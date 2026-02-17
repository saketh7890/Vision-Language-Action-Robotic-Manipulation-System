#!/usr/bin/env python3
"""
LLM-POWERED DOBOT PICK & PLACE SYSTEM – v4.3

High-level view
---------------
This script ties together three layers:

  1. Perception
     - Uses a USB camera and simple HSV color thresholds to detect
       colored blocks (red, blue, green, yellow) in image space.

  2. World state (memory)
     - Maintains a persistent logical model of blocks, their
       positions in pixel space, any stacks that exist, and what
       the gripper is holding.
     - This state is saved to and loaded from disk between runs.

  3. Control / Planning
     - A Dobot Magician Lite is controlled via pydobot.
     - A Groq-hosted LLM generates *Python code* that uses a small,
       whitelisted API (dobot + system_state methods + time.sleep).
     - A validator ensures the generated code is safe before execution.

Version 4.3 focuses on:
  - Robust handling of multiple blocks of the same color.
  - Persistent memory of:
        * block positions (pixel coordinates),
        * whether they are stacked or not,
        * which blocks (and colors) are stacked together.
  - Safer stack bookkeeping when picking / placing:
        * Picking a block removes it from the stack at that location.
        * Placing a block adds it to the stack at the new location.
  - A structured LLM prompt so natural-language commands such as:
        * "pick the farthest red block and place it on the nearest green block"
        * "place the blue block right next to the stack"
        * "place the yellow block over the stack"
    are interpreted in a consistent, state-aware way.

Behavior is unchanged from the original version; this is mainly a
clarified, slightly re-commented interpretation of the same logic.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore
from langchain_groq import ChatGroq  # type: ignore
from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore


__version__ = "4.3.0"

# ---------------------------------------------------------------------------
# Configuration / constants
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)
if not GROQ_API_KEY:
    print("[WARN] GROQ_API_KEY is not set. Natural-language control will be disabled.")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY or ""

DOBOT_PORT = os.environ.get("DOBOT_PORT", "/dev/ttyACM0")
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "4"))

# Camera frame (used for bounds clamping when computing adjacent placements)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# State file for persistent memory
STATE_FILE = Path(os.environ.get("DOBOT_STATE_FILE", str(Path.home() / "dobot_state.json")))

# Dobot Positions
HOME_POSITION = {'x': 250.0, 'y': 0.0, 'z': 180.0, 'r': 0.0}

# Z-heights (you may need to tweak these for your table/blocks)
PICK_HEIGHT = -44.0
SAFE_HEIGHT = 50.0
PLACE_HEIGHT = -44.0
BLOCK_HEIGHT = 17.9  # mm vertical spacing between stacked blocks

# In-plane block spacing
BLOCK_SIDE_MM = 15.0
MIN_NEXT_TO_CLEARANCE_MM = BLOCK_SIDE_MM

# Optional fine-tuning offsets in mm (for camera→robot misalignment)
PICK_X_OFFSET_MM = float(os.environ.get("PICK_X_OFFSET_MM", "0"))
PICK_Y_OFFSET_MM = float(os.environ.get("PICK_Y_OFFSET_MM", "0"))
PICK_Z_OFFSET_MM = float(os.environ.get("PICK_Z_OFFSET_MM", "0"))
PLACE_X_OFFSET_MM = float(os.environ.get("PLACE_X_OFFSET_MM", "0"))
PLACE_Y_OFFSET_MM = float(os.environ.get("PLACE_Y_OFFSET_MM", "0"))
PLACE_Z_OFFSET_MM = float(os.environ.get("PLACE_Z_OFFSET_MM", "0"))

# Calibration points (pixel → dobot XY)
CALIBRATION_POINTS = [
    {'pixel': (139, 429), 'dobot': (241.120, 74.051)},
    {'pixel': (142, 246), 'dobot': (332.479, 75.292)},
    {'pixel': (313, 361), 'dobot': (275.933, -11.462)},
    {'pixel': (508, 436), 'dobot': (239.647, -110.390)},
    {'pixel': (505, 257), 'dobot': (330.925, -108.923)},
]


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

class CoordinateConverter:
    """
    Converts pixel coordinates from the camera image into Dobot XY
    coordinates using a 2D affine transformation.

    The CALIBRATION_POINTS list at the top of the file defines the mapping
    from image pixel coordinates to real-world Dobot coordinates.

    Once the affine parameters are estimated, this class also exposes
    approximate mm-per-pixel scales along x and y. These are later used
    to compute safe "next to" positions in pixel space while reasoning
    about distances in millimeters.
    """

    def __init__(self) -> None:
        if len(CALIBRATION_POINTS) < 3:
            raise ValueError("Need at least 3 calibration points for affine transform.")
        self.calibration_points = CALIBRATION_POINTS
        self.avg_r = 14.25  # default wrist rotation
        self.calculate_transformation()

        # Derived mm-per-pixel scale estimates (for "next to" spacing etc.)
        self.mm_per_px_x = float(np.sqrt(self.a11 ** 2 + self.a21 ** 2))
        self.mm_per_px_y = float(np.sqrt(self.a12 ** 2 + self.a22 ** 2))
        print(f"[Coordinates] ✓ Calibration loaded (mm/px: x≈{self.mm_per_px_x:.3f}, y≈{self.mm_per_px_y:.3f})")

    def calculate_transformation(self) -> None:
        """Calculate affine transformation matrix from calibration points."""
        pixel_coords = np.array([
            [p['pixel'][0], p['pixel'][1], 1] for p in self.calibration_points
        ])
        dobot_x = np.array([p['dobot'][0] for p in self.calibration_points])
        dobot_y = np.array([p['dobot'][1] for p in self.calibration_points])

        coeffs_x, _, _, _ = np.linalg.lstsq(pixel_coords, dobot_x, rcond=None)
        self.a11, self.a12, self.b1 = coeffs_x

        coeffs_y, _, _, _ = np.linalg.lstsq(pixel_coords, dobot_y, rcond=None)
        self.a21, self.a22, self.b2 = coeffs_y

        print("[Coordinates] Transformation coefficients calculated")

    def pixel_to_dobot(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """Convert pixel coordinates to Dobot XY coordinates."""
        dobot_x = self.a11 * pixel_x + self.a12 * pixel_y + self.b1
        dobot_y = self.a21 * pixel_x + self.a22 * pixel_y + self.b2
        print(f"[Coord] Pixel ({pixel_x}, {pixel_y}) → Dobot ({dobot_x:.1f}, {dobot_y:.1f})")
        return float(dobot_x), float(dobot_y)

    def mm_to_pixels(self, dx_mm: float = 0.0, dy_mm: float = 0.0, axis: str = 'x') -> int:
        """
        Convert a desired delta in millimeters into a pixel offset along
        the x or y axis of the image.

        This is approximate and based purely on the calibration-derived
        scale; it is sufficient for computing "next to" placements.
        """
        if axis == 'x':
            if self.mm_per_px_x <= 1e-6:
                return 0
            return int(round(dx_mm / self.mm_per_px_x))
        else:
            if self.mm_per_px_y <= 1e-6:
                return 0
            return int(round(dy_mm / self.mm_per_px_y))


# ---------------------------------------------------------------------------
# System state tracking
# ---------------------------------------------------------------------------

class SystemState:
    """
    Central, persistent representation of the world as understood by this
    script. It does not talk directly to hardware.

    For each block we remember:
      - color
      - per-color id (1, 2, 3, ...)
      - pixel position (x, y)
      - whether it's on the table / in a stack / in the gripper
      - which blocks are below it (for stacks)

    Additionally, the SystemState maintains:
      - stack_map : maps coarse pixel "locations" to a list of block_ids,
                    ordered bottom → top.
      - stack_meta: cached metadata about each stack (centroid, height).
      - conversation_history: a small log of recent commands.
      - gripper_status / holding_block: whether we currently hold something.
    """

    def __init__(self, state_file: Path = STATE_FILE, coord_converter: Optional[CoordinateConverter] = None) -> None:
        self.state_file = state_file
        self.gripper_status: str = "EMPTY"
        self.holding_block: Optional[str] = None
        self.blocks: Dict[str, Dict[str, Any]] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.stack_map: Dict[str, List[str]] = {}  # {location_key: [block_ids bottom→top]}
        self.stack_meta: Dict[str, Dict[str, Any]] = {}  # {location_key: {'x':px, 'y':px, 'height':int, 'updated_at':iso}}
        self.last_place_location: Optional[Dict[str, Any]] = None
        self.last_manipulated_block_id: Optional[str] = None  # "that"/"it" resolver
        self.min_adjacent_clearance_mm: float = MIN_NEXT_TO_CLEARANCE_MM

        self.coord_converter: CoordinateConverter = coord_converter or CoordinateConverter()

    # ----------------- stack helper internals -----------------

    def _location_key(self, pixel_x: int, pixel_y: int, tolerance: int = 30) -> str:
        """Create location key for grouping blocks at same position (bucketing by ~tolerance px)."""
        x_bucket = round(pixel_x / tolerance) * tolerance
        y_bucket = round(pixel_y / tolerance) * tolerance
        return f"{x_bucket},{y_bucket}"

    def _update_stack_meta_for_key(self, loc_key: str) -> None:
        """Update centroid (avg pixel) and height for a stack location key."""
        if loc_key not in self.stack_map or len(self.stack_map[loc_key]) == 0:
            if loc_key in self.stack_meta:
                del self.stack_meta[loc_key]
            return

        bids = self.stack_map[loc_key]
        xs = [self.blocks[bid]['pixel_x'] for bid in bids if bid in self.blocks]
        ys = [self.blocks[bid]['pixel_y'] for bid in bids if bid in self.blocks]
        if not xs or not ys:
            return

        cx = int(round(sum(xs) / len(xs)))
        cy = int(round(sum(ys) / len(ys)))
        self.stack_meta[loc_key] = {
            'x': cx,
            'y': cy,
            'height': len(bids),
            'updated_at': datetime.now().isoformat()
        }

    def rebuild_stacks(self) -> None:
        """
        Rebuild stack_map and stack_meta purely from the per-block data.

        Only blocks that are NOT in the gripper are considered. This is the
        authoritative way to recompute what stacks exist and how tall they are.
        """
        self.stack_map = {}
        self.stack_meta = {}

        for block_id, block in self.blocks.items():
            if block.get('in_gripper'):
                continue
            loc_key = self._location_key(block['pixel_x'], block['pixel_y'])
            self.stack_map.setdefault(loc_key, []).append(block_id)

        # Sort each stack by known stack_level (if available)
        for loc_key, ids in self.stack_map.items():
            self.stack_map[loc_key] = sorted(
                ids,
                key=lambda bid: self.blocks.get(bid, {}).get('stack_level', 0)
            )
            self._update_stack_meta_for_key(loc_key)

    # ----------------- block initialization / updates -----------------

    def initialize_blocks(self, detected_blocks: List[Dict[str, Any]]) -> None:
        """Initialize blocks from a fresh camera detection snapshot."""
        self.blocks = {}
        self.stack_map = {}
        self.stack_meta = {}

        for block in detected_blocks:
            block_id = block['global_id']
            self.blocks[block_id] = {
                'color': block['color'],
                'id': block['id'],
                'pixel_x': block['pixel_x'],
                'pixel_y': block['pixel_y'],
                'original_x': block['pixel_x'],
                'original_y': block['pixel_y'],
                'current_position': {'x': block['pixel_x'], 'y': block['pixel_y'], 'z': 0.0},
                'stack_level': 0,  # 0 = on table
                'blocks_below': [],
                'in_gripper': False
            }

        self.rebuild_stacks()
        print(f"[State] Initialized {len(self.blocks)} blocks")
        self.save_state()

    def update_gripper(self, block_id: Optional[str] = None) -> None:
        """
        Update gripper status + in_gripper flags + stack membership.

        Convention:
        - When called with a block_id: you have just picked that block.
          → mark it in_gripper=True and remove it from any stack it belonged to.
        - When called with None: you just released the currently held block.
          → its in_gripper flag will be cleared when you call update_block_position
            for the place location; here we just mark gripper as empty.
        """
        if block_id is None:
            # Releasing whatever we were holding
            print("[State] Gripper: now EMPTY")
            self.gripper_status = "EMPTY"
            # We DO NOT clear in_gripper here; update_block_position will handle it
            self.holding_block = None
        else:
            # Picking this block
            self.gripper_status = "HOLDING"
            self.holding_block = block_id
            self.last_manipulated_block_id = block_id

            block = self.blocks.get(block_id)
            if block:
                block['in_gripper'] = True
                loc_key = self._location_key(block['pixel_x'], block['pixel_y'])
                if loc_key in self.stack_map and block_id in self.stack_map[loc_key]:
                    self.stack_map[loc_key].remove(block_id)
                    if not self.stack_map[loc_key]:
                        del self.stack_map[loc_key]
                        if loc_key in self.stack_meta:
                            del self.stack_meta[loc_key]
                    else:
                        self._update_stack_meta_for_key(loc_key)

        print(f"[State] Gripper status: {self.gripper_status}, holding: {self.holding_block}")
        self.save_state()

    def update_block_position(
        self,
        block_id: str,
        pixel_x: int,
        pixel_y: int,
        z: float = 0.0,
        blocks_below: Optional[List[str]] = None
    ) -> None:
        """
        Update block position with stack awareness.

        This is called after placing a block. It:
        - sets new pixel & z,
        - adds it to the stack at that location,
        - records which blocks are below,
        - marks in_gripper=False.
        """
        if block_id not in self.blocks:
            print(f"[State] ⚠️ update_block_position called on unknown block_id={block_id}")
            return

        block = self.blocks[block_id]

        # Remove from old stack location if present
        old_loc_key = self._location_key(block['pixel_x'], block['pixel_y'])
        if old_loc_key in self.stack_map and block_id in self.stack_map[old_loc_key]:
            self.stack_map[old_loc_key].remove(block_id)
            if not self.stack_map[old_loc_key]:
                del self.stack_map[old_loc_key]
                if old_loc_key in self.stack_meta:
                    del self.stack_meta[old_loc_key]
            else:
                self._update_stack_meta_for_key(old_loc_key)

        # Update position
        block['pixel_x'] = pixel_x
        block['pixel_y'] = pixel_y
        block['current_position'] = {'x': pixel_x, 'y': pixel_y, 'z': z}
        block['in_gripper'] = False

        # Calculate stack level from z offset (0 if on table)
        stack_level = round(z / BLOCK_HEIGHT) if z > 0 else 0
        block['stack_level'] = stack_level

        # Update blocks_below list
        block['blocks_below'] = (blocks_below or [])

        # Add to new stack location
        new_loc_key = self._location_key(pixel_x, pixel_y)
        self.stack_map.setdefault(new_loc_key, [])
        if block_id not in self.stack_map[new_loc_key]:
            self.stack_map[new_loc_key].append(block_id)

        # Sort by z level and refresh meta
        self.stack_map[new_loc_key].sort(
            key=lambda bid: self.blocks[bid]['stack_level']
        )
        self._update_stack_meta_for_key(new_loc_key)

        # Track last place location + last manipulated
        self.last_place_location = {'x': pixel_x, 'y': pixel_y, 'z': z}
        self.last_manipulated_block_id = block_id

        print(f"[State] {block_id} moved to ({pixel_x}, {pixel_y}, z={z}, level={stack_level})")
        self.save_state()

    # ----------------- state queries -----------------

    def get_stack_at_location(self, pixel_x: int, pixel_y: int) -> List[str]:
        """Get stack of blocks at a location (list bottom→top)."""
        loc_key = self._location_key(pixel_x, pixel_y)
        return self.stack_map.get(loc_key, [])

    def get_stack_height(self, pixel_x: int, pixel_y: int) -> int:
        """Get number of blocks stacked at location."""
        return len(self.get_stack_at_location(pixel_x, pixel_y))

    def get_block(self, block_id: str) -> Optional[Dict[str, Any]]:
        """Get block data by ID."""
        return self.blocks.get(block_id)

    def is_gripper_empty(self) -> bool:
        return self.gripper_status == "EMPTY"

    def get_holding_block_id(self) -> Optional[str]:
        return self.holding_block

    def get_last_manipulated_block_id(self) -> Optional[str]:
        """Use for pronouns like 'that' or 'it' referring to the previously handled block."""
        return self.last_manipulated_block_id

    def _distance_to_robot_home_mm(self, pixel_x: int, pixel_y: int) -> float:
        """Euclidean distance in mm from HOME_POSITION in Dobot XY space."""
        dx, dy = self.coord_converter.pixel_to_dobot(pixel_x, pixel_y)
        return float(np.hypot(dx - HOME_POSITION['x'], dy - HOME_POSITION['y']))

    def _nearest_or_farthest_block(self, block_ids: List[str], mode: str = 'nearest') -> Optional[str]:
        # Ignore blocks currently in the gripper
        block_ids = [bid for bid in block_ids if not self.blocks.get(bid, {}).get('in_gripper')]
        if not block_ids:
            return None
        scored: List[Tuple[float, str]] = []
        for bid in block_ids:
            b = self.blocks.get(bid)
            if not b:
                continue
            d = self._distance_to_robot_home_mm(b['pixel_x'], b['pixel_y'])
            scored.append((d, bid))
        if not scored:
            return None
        scored.sort(key=lambda t: t[0])
        return scored[0][1] if mode == 'nearest' else scored[-1][1]

    def get_nearest_block_id_by_color(self, color: str) -> Optional[str]:
        ids = [bid for bid, b in self.blocks.items()
               if b['color'].lower() == color.lower() and not b.get('in_gripper')]
        return self._nearest_or_farthest_block(ids, mode='nearest')

    def get_farthest_block_id_by_color(self, color: str) -> Optional[str]:
        ids = [bid for bid, b in self.blocks.items()
               if b['color'].lower() == color.lower() and not b.get('in_gripper')]
        return self._nearest_or_farthest_block(ids, mode='farthest')

    def get_nearest_block_id(self) -> Optional[str]:
        ids = [bid for bid, b in self.blocks.items() if not b.get('in_gripper')]
        return self._nearest_or_farthest_block(ids, mode='nearest')

    def get_farthest_block_id(self) -> Optional[str]:
        ids = [bid for bid, b in self.blocks.items() if not b.get('in_gripper')]
        return self._nearest_or_farthest_block(ids, mode='farthest')

    def get_stack_locations(self, min_height: int = 2) -> List[Dict[str, Any]]:
        """
        Return list of dicts for each stack with height >= min_height:
        [{'loc_key': str, 'x': int, 'y': int, 'height': int}, ...], tallest first.
        """
        out: List[Dict[str, Any]] = []
        for loc_key, meta in self.stack_meta.items():
            h = int(meta.get('height', 0))
            if h >= min_height:
                out.append({'loc_key': loc_key, 'x': int(meta['x']), 'y': int(meta['y']), 'height': h})
        out.sort(key=lambda d: d['height'], reverse=True)
        return out

    def get_tallest_stack_xy(self, min_height: int = 2) -> Optional[Tuple[int, int, int]]:
        """Return (x, y, height) for tallest stack (height>=2). None if no such stack."""
        stacks = self.get_stack_locations(min_height=min_height)
        if not stacks:
            return None
        s = stacks[0]
        return int(s['x']), int(s['y']), int(s['height'])

    def get_nearest_stack_xy(self, min_height: int = 2) -> Optional[Tuple[int, int, int]]:
        """Return (x, y, height) for nearest stack to robot. None if no stack."""
        stacks = self.get_stack_locations(min_height=min_height)
        if not stacks:
            return None
        best = None
        bestd = None
        for s in stacks:
            d = self._distance_to_robot_home_mm(int(s['x']), int(s['y']))
            if best is None or d < bestd:  # type: ignore
                best = s
                bestd = d
        assert best is not None
        return int(best['x']), int(best['y']), int(best['height'])

    # ----------------- "next to" helpers -----------------

    def compute_adjacent_position(
        self,
        ref_x: int,
        ref_y: int,
        direction: str = 'right',
        clearance_mm: Optional[float] = None
    ) -> Tuple[int, int]:
        """
        Compute a pixel position adjacent to (ref_x, ref_y) with at least
        `clearance_mm` center-to-center spacing.

        Directions:
          - 'right' (+x),
          - 'left'  (-x),
          - 'front' (-y, toward top of image),
          - 'back'  (+y, toward bottom of image).

        Coordinates are clamped to the camera frame bounds.
        """
        base_clearance = self.min_adjacent_clearance_mm if clearance_mm is None else float(clearance_mm)
        clr = base_clearance + 3.0  # 3mm safety margin

        dx_px = 0
        dy_px = 0

        if direction == 'right':
            dx_px = self.coord_converter.mm_to_pixels(dx_mm=clr, axis='x')
        elif direction == 'left':
            dx_px = -self.coord_converter.mm_to_pixels(dx_mm=clr, axis='x')
        elif direction == 'front':
            dy_px = -self.coord_converter.mm_to_pixels(dy_mm=clr, axis='y')
        elif direction == 'back':
            dy_px = self.coord_converter.mm_to_pixels(dy_mm=clr, axis='y')
        else:
            dx_px = self.coord_converter.mm_to_pixels(dx_mm=clr, axis='x')

        nx = int(np.clip(ref_x + dx_px, 0, FRAME_WIDTH - 1))
        ny = int(np.clip(ref_y + dy_px, 0, FRAME_HEIGHT - 1))
        print(f"[State] Adjacent ({direction}, ≥{clr}mm): ({ref_x},{ref_y}) → ({nx},{ny}) [dx_px={dx_px}, dy_px={dy_px}]")
        return nx, ny

    def compute_adjacent_to_block(
        self,
        block_id: str,
        direction: str = 'right',
        clearance_mm: Optional[float] = None
    ) -> Optional[Tuple[int, int]]:
        """Adjacent pixel (x,y) next to the given block id."""
        b = self.blocks.get(block_id)
        if not b:
            print(f"[State] ⚠️ compute_adjacent_to_block: unknown block_id={block_id}")
            return None
        if b.get('in_gripper'):
            print(f"[State] ⚠️ compute_adjacent_to_block: {block_id} is in gripper; using last known table position")
        return self.compute_adjacent_position(b['pixel_x'], b['pixel_y'], direction=direction, clearance_mm=clearance_mm)

    def compute_adjacent_to_stack(
        self,
        direction: str = 'right',
        strategy: str = 'nearest',
        clearance_mm: Optional[float] = None
    ) -> Optional[Tuple[int, int]]:
        """
        Adjacent pixel (x, y) next to a chosen stack (height ≥ 2).

        strategy:
          - 'nearest' : stack closest to robot
          - 'tallest' : highest stack
        """
        if strategy == 'tallest':
            sel = self.get_tallest_stack_xy(min_height=2)
        else:
            sel = self.get_nearest_stack_xy(min_height=2)
        if not sel:
            print("[State] ⚠️ compute_adjacent_to_stack: no stacks (height>=2) available")
            return None
        sx, sy, _ = sel
        return self.compute_adjacent_position(sx, sy, direction=direction, clearance_mm=clearance_mm)

    # ----------------- conversation + persistence -----------------

    def add_to_history(self, prompt: str, action: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Add command to history with optional details."""
        entry = {
            'prompt': prompt,
            'action': action,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.conversation_history.append(entry)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        self.save_state()

    def get_state_summary(self) -> str:
        """
        Get a formatted snapshot of the current state.

        This is primarily used inside the LLM system prompt, but it is
        also useful for debugging when printed via the 'status' command.
        """
        summary = f"""CURRENT SYSTEM STATE:
- Gripper Status: {self.gripper_status}
- Holding Block: {self.holding_block or 'None'}
- Last Manipulated Block: {self.last_manipulated_block_id or 'None'}

AVAILABLE BLOCKS (with stacking info):
"""
        for block_id, block in self.blocks.items():
            stack_info = ""
            if block.get('in_gripper'):
                stack_info = " [IN GRIPPER]"
            elif block['stack_level'] > 0:
                stack_info = f" [STACKED - Level {block['stack_level']}, on: {', '.join(block['blocks_below'])}]"
            summary += f"  - {block_id}: {block['color']} at pixel ({block['pixel_x']}, {block['pixel_y']}){stack_info}\n"

        # Show stacks
        if self.stack_map:
            summary += "\nDETECTED STACKS:\n"
            for loc_key, stack_blocks in self.stack_map.items():
                if len(stack_blocks) > 1:
                    meta = self.stack_meta.get(loc_key, {})
                    cx = meta.get('x', '?')
                    cy = meta.get('y', '?')
                    stack_descr = " → ".join(
                        f"{bid}({self.blocks[bid]['color']})" for bid in stack_blocks if bid in self.blocks
                    )
                    summary += f"  - Location {loc_key} @ ({cx},{cy}): {stack_descr} ({len(stack_blocks)} high)\n"

        # Show last place location
        if self.last_place_location:
            summary += f"\nLAST PLACE LOCATION: pixel ({self.last_place_location['x']}, {self.last_place_location['y']}, z={self.last_place_location['z']})\n"

        # Show recent actions
        if self.conversation_history:
            summary += f"\nRECENT ACTIONS (last {min(5, len(self.conversation_history))}):\n"
            for entry in self.conversation_history[-5:]:
                details_str = ""
                if entry.get('details'):
                    details_str = f" | {entry['details']}"
                summary += f"  - '{entry['prompt']}' → {entry['action']}{details_str}\n"

        return summary

    def save_state(self) -> None:
        """Save state to JSON file."""
        try:
            state_data = {
                'gripper_status': self.gripper_status,
                'holding_block': self.holding_block,
                'blocks': self.blocks,
                'conversation_history': self.conversation_history,
                'stack_map': self.stack_map,
                'stack_meta': self.stack_meta,
                'last_place_location': self.last_place_location,
                'last_manipulated_block_id': self.last_manipulated_block_id,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            print(f"[State] ✓ Saved to {self.state_file}")
        except Exception as e:
            print(f"[State] ⚠️ Save failed: {e}")

    def load_state(self) -> bool:
        """Load state from JSON file."""
        try:
            if not self.state_file.exists():
                print(f"[State] No saved state found at {self.state_file}")
                return False
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
            self.gripper_status = state_data.get('gripper_status', 'EMPTY')
            self.holding_block = state_data.get('holding_block')
            self.blocks = state_data.get('blocks', {})
            self.conversation_history = state_data.get('conversation_history', [])
            self.stack_map = state_data.get('stack_map', {})
            self.stack_meta = state_data.get('stack_meta', {})
            self.last_place_location = state_data.get('last_place_location')
            self.last_manipulated_block_id = state_data.get('last_manipulated_block_id')
            print(f"[State] ✓ Loaded state from {self.state_file}")
            print(f"[State] Restored: {len(self.blocks)} blocks, {len(self.conversation_history)} history entries")

            # Ensure newer field exists
            for block in self.blocks.values():
                block.setdefault('in_gripper', False)
                block.setdefault('stack_level', 0)
                block.setdefault('blocks_below', [])
                block.setdefault('current_position', {
                    'x': block.get('pixel_x', 0),
                    'y': block.get('pixel_y', 0),
                    'z': 0.0
                })

            # Rebuild stacks to be safe
            self.rebuild_stacks()

            return True
        except Exception as e:
            print(f"[State] ⚠️ Load failed: {e}")
            return False


# ---------------------------------------------------------------------------
# Dobot controller
# ---------------------------------------------------------------------------

class DobotController:
    """Hardware control for Dobot Magician Lite."""

    def __init__(self, port: str = DOBOT_PORT, coord_converter: Optional[CoordinateConverter] = None) -> None:
        self.port = port
        self.device = None
        self.coord_converter: CoordinateConverter = coord_converter or CoordinateConverter()

    def connect(self, port: Optional[str] = None) -> bool:
        """Connect to Dobot."""
        port = port or self.port
        try:
            from pydobot import Dobot  # type: ignore
        except Exception as e:
            print(f"[Dobot] ✗ ERROR: pydobot not available ({e})")
            return False

        try:
            print(f"\n[Dobot] Connecting to {port}...")
            self.device = Dobot(port=port)
            time.sleep(1)
            if hasattr(self.device, 'set_speed'):
                self.device.set_speed(velocity=150, acceleration=150)
            print("[Dobot] ✓ Connected successfully!")
            return True
        except Exception as e:
            print(f"[Dobot] ✗ ERROR: {e}")
            self.device = None
            return False

    def move_to(self, x: float, y: float, z: float, r: float = 0.0, wait: bool = True) -> None:
        """Move Dobot to coordinates."""
        if self.device:
            self.device.move_to(x, y, z, r, wait=wait)
            if wait:
                time.sleep(0.2)

    def go_home(self) -> None:
        """Move to home position."""
        print("[Dobot] Moving to home position...")
        self.move_to(HOME_POSITION['x'], HOME_POSITION['y'], HOME_POSITION['z'], HOME_POSITION['r'])
        print("[Dobot] ✓ At home position")

    def suction_on(self) -> None:
        """Enable suction cup."""
        if self.device:
            self.device.suck(True)
            print("[Dobot] Suction: ON")
        time.sleep(0.8)

    def suction_off(self) -> None:
        """Disable suction cup."""
        if self.device:
            self.device.suck(False)
            print("[Dobot] Suction: OFF")
        time.sleep(0.5)

    def pick_block(self, pixel_x: int, pixel_y: int, z_offset: float = 0.0) -> None:
        """Pick up block at pixel coordinates, with optional z_offset for stacked blocks."""
        print(f"\n[Dobot] Picking block at pixel ({pixel_x}, {pixel_y}) (z_offset={z_offset:.1f}mm)...")
        dobot_x, dobot_y = self.coord_converter.pixel_to_dobot(pixel_x, pixel_y)
        dobot_x += PICK_X_OFFSET_MM
        dobot_y += PICK_Y_OFFSET_MM
        target_z = PICK_HEIGHT + PICK_Z_OFFSET_MM + z_offset
        print(f"[Dobot] Pick pose: x={dobot_x:.1f}, y={dobot_y:.1f}, z={target_z:.1f}")
        self.move_to(dobot_x, dobot_y, SAFE_HEIGHT)
        self.move_to(dobot_x, dobot_y, target_z)
        self.suction_on()
        self.move_to(dobot_x, dobot_y, SAFE_HEIGHT)
        print("[Dobot] ✓ Block picked")

    def place_block(self, pixel_x: int, pixel_y: int, z_offset: float = 0.0) -> None:
        """Place block at pixel coordinates with optional Z offset for stacking."""
        print(f"\n[Dobot] Placing block at pixel ({pixel_x}, {pixel_y}), z_offset={z_offset:.1f}mm...")
        dobot_x, dobot_y = self.coord_converter.pixel_to_dobot(pixel_x, pixel_y)
        dobot_x += PLACE_X_OFFSET_MM
        dobot_y += PLACE_Y_OFFSET_MM
        place_z = PLACE_HEIGHT + PLACE_Z_OFFSET_MM + z_offset
        print(f"[Dobot] Place pose: x={dobot_x:.1f}, y={dobot_y:.1f}, z={place_z:.1f}")
        self.move_to(dobot_x, dobot_y, SAFE_HEIGHT)
        self.move_to(dobot_x, dobot_y, place_z)
        self.suction_off()
        self.move_to(dobot_x, dobot_y, SAFE_HEIGHT)
        print("[Dobot] ✓ Block placed")

    def close(self) -> None:
        """Close Dobot connection."""
        if self.device:
            try:
                self.device.close()
                print("[Dobot] Connection closed")
            except Exception:
                pass
            finally:
                self.device = None


# ---------------------------------------------------------------------------
# Block detection
# ---------------------------------------------------------------------------

class BlockDetector:
    """Camera-based block detection."""

    def __init__(self, camera_index: int = CAMERA_INDEX) -> None:
        self.camera_index = camera_index
        self.camera = None
        # HSV ranges; tweak per camera if needed
        self.color_ranges = {
            'red': [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255]))
            ],
            'blue': [(np.array([103, 140, 60]), np.array([130, 255, 255]))],
            'green': [(np.array([40, 100, 100]), np.array([80, 255, 255]))],
            'yellow': [(np.array([20, 100, 100]), np.array([35, 255, 255]))]
        }

    def detect_color_blocks(self, frame: np.ndarray, color_name: str) -> List[Dict[str, Any]]:
        """Detect blocks of a specific color in a frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = None
        for lower, upper in self.color_ranges[color_name]:
            if mask is None:
                mask = cv2.inRange(hsv, lower, upper)
            else:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected: List[Dict[str, Any]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 10000:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    detected.append({'center': (cx, cy), 'contour': contour, 'area': area})
        return detected

    def detect_all_blocks_live(self) -> Optional[List[Dict[str, Any]]]:
        """Open camera and detect all blocks in real-time (manual spacebar capture)."""
        print(f"\n[Camera] Opening camera {self.camera_index}...")
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                print("[Camera] ✗ ERROR: Failed to open camera")
                return None
            print("[Camera] ✓ Camera opened successfully")
        except Exception as e:
            print(f"[Camera] ✗ ERROR: {e}")
            return None

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        print("[Camera] Press SPACE to capture | Q to quit")
        detected_blocks: Optional[List[Dict[str, Any]]] = []

        while True:
            ret, frame = self.camera.read()
            if not ret:
                break

            display = frame.copy()
            all_detected: List[Dict[str, Any]] = []
            for color_name in ['red', 'blue', 'green', 'yellow']:
                blocks = self.detect_color_blocks(frame, color_name)
                for block in blocks:
                    all_detected.append({'color': color_name, 'center': block['center'], 'contour': block['contour']})

            # Sorting is purely for consistent labeling
            all_detected.sort(key=lambda b: (b['center'][1] // 50, b['center'][0]))

            vis_colors = {'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0), 'yellow': (0, 255, 255)}
            temp_blocks: List[Dict[str, Any]] = []
            temp_color_counters = {'red': 1, 'blue': 1, 'green': 1, 'yellow': 1}

            for block in all_detected:
                cx, cy = block['center']
                color_name = block['color']
                color_bgr = vis_colors[color_name]
                color_id = temp_color_counters[color_name]
                temp_color_counters[color_name] += 1

                cv2.drawContours(display, [block['contour']], -1, color_bgr, 2)
                cv2.circle(display, (cx, cy), 5, color_bgr, -1)

                label = f"{color_name}_{color_id}"
                cv2.putText(display, label, (cx - 30, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display, f"({cx},{cy})", (cx - 30, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                temp_blocks.append({'color': color_name, 'id': color_id, 'center': (cx, cy)})

            cv2.putText(display, f"Blocks: {len(all_detected)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Block Detection", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if temp_blocks:
                    detected_blocks = []
                    for block in temp_blocks:
                        global_id = f"{block['color']}_{block['id']}"
                        detected_blocks.append({
                            'global_id': global_id,
                            'color': block['color'],
                            'id': block['id'],
                            'pixel_x': block['center'][0],
                            'pixel_y': block['center'][1]
                        })
                    print(f"\n[Camera] ✓ Captured {len(detected_blocks)} blocks")
                    break
            elif key == ord('q'):
                print("[Camera] Detection cancelled")
                detected_blocks = None
                break

        self.camera.release()
        cv2.destroyAllWindows()
        return detected_blocks

    def auto_capture_blocks(self, display_duration: float = 2.0) -> Optional[List[Dict[str, Any]]]:
        """Auto-capture blocks after showing live feed for a short duration."""
        print("\n[Camera] Auto-capturing block positions...")
        try:
            camera = cv2.VideoCapture(self.camera_index)
            if not camera.isOpened():
                print("[Camera] ✗ ERROR: Failed to open camera")
                return None
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

            vis_colors = {'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0), 'yellow': (0, 255, 255)}
            start_time = time.time()
            latest_blocks: List[Dict[str, Any]] = []

            while True:
                ret, frame = camera.read()
                if not ret:
                    break
                display = frame.copy()
                all_detected: List[Dict[str, Any]] = []
                for color_name in ['red', 'blue', 'green', 'yellow']:
                    blocks = self.detect_color_blocks(frame, color_name)
                    for block in blocks:
                        all_detected.append({'color': color_name, 'center': block['center'], 'contour': block['contour']})
                all_detected.sort(key=lambda b: (b['center'][1] // 50, b['center'][0]))

                temp_color_counters = {'red': 1, 'blue': 1, 'green': 1, 'yellow': 1}
                temp_blocks: List[Dict[str, Any]] = []
                for block in all_detected:
                    cx, cy = block['center']
                    color_name = block['color']
                    color_bgr = vis_colors[color_name]
                    color_id = temp_color_counters[color_name]
                    temp_color_counters[color_name] += 1

                    cv2.drawContours(display, [block['contour']], -1, color_bgr, 2)
                    cv2.circle(display, (cx, cy), 5, color_bgr, -1)

                    label = f"{color_name}_{color_id}"
                    cv2.putText(display, label, (cx - 30, cy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    temp_blocks.append({
                        'global_id': label,
                        'color': color_name,
                        'id': color_id,
                        'pixel_x': cx,
                        'pixel_y': cy
                    })

                latest_blocks = temp_blocks
                elapsed = time.time() - start_time
                cv2.putText(display, f"AUTO-CAPTURE - Blocks: {len(all_detected)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Auto Block Capture", display)
                cv2.waitKey(30)
                if elapsed >= display_duration:
                    break

            camera.release()
            cv2.destroyAllWindows()
            time.sleep(0.2)
            return latest_blocks
        except Exception as e:
            print(f"[Camera] ✗ ERROR: {e}")
            return None


# ---------------------------------------------------------------------------
# LLM code generator
# ---------------------------------------------------------------------------

class LLMCodeGenerator:
    """
    LLM wrapper that turns natural language into safe Python code
    for this system.

    It does not execute the code or validate it; that is delegated to
    CodeExecutor. This class is responsible for:
      - assembling a rich system prompt (including the state summary),
      - sending the request to the Groq-hosted model,
      - stripping away any accidental markdown fences.
    """

    def __init__(self, api_key: Optional[str]) -> None:
        self.enabled = bool(api_key)
        self.llm: Optional[ChatGroq] = None

        if self.enabled:
            self.llm = ChatGroq(
                temperature=0.1,
                model_name="llama-3.3-70b-versatile",
                api_key=api_key or ""
            )
        else:
            print("[LLM] Disabled (no GROQ_API_KEY); natural-language commands will not work.")

        self.system_prompt_template = """You are a contextually-aware Python code generator for a Dobot robotic arm.

{state_summary}

ALLOWED FUNCTIONS:
- dobot.pick_block(pixel_x, pixel_y, z_offset=0)
- dobot.place_block(pixel_x, pixel_y, z_offset=0)
- system_state.update_gripper(block_id or None)
- system_state.update_block_position(block_id, pixel_x, pixel_y, z, blocks_below=[...])
- system_state.get_block(block_id)
- system_state.get_stack_at_location(pixel_x, pixel_y)
- system_state.get_stack_height(pixel_x, pixel_y)
- system_state.is_gripper_empty()
- system_state.get_holding_block_id()
- system_state.get_last_manipulated_block_id()
- system_state.get_nearest_block_id_by_color(color_name)
- system_state.get_farthest_block_id_by_color(color_name)
- system_state.get_nearest_block_id()
- system_state.get_farthest_block_id()
- system_state.get_tallest_stack_xy()
- system_state.get_nearest_stack_xy()
- system_state.compute_adjacent_position(x, y, direction='right'|'left'|'front'|'back', clearance_mm=15.0)
- system_state.compute_adjacent_to_block(block_id, direction='right'|'left'|'front'|'back', clearance_mm=15.0)
- system_state.compute_adjacent_to_stack(direction='right'|'left'|'front'|'back', strategy='nearest'|'tallest', clearance_mm=15.0)
- time.sleep(seconds)

RULES:
- OUTPUT ONLY PYTHON CODE. No explanations, no comments, no markdown fences.
- Use ONLY the allowed functions and Python builtins (print, len, range, int, float, str, round).

SEMANTICS & INTERPRETATION RULES:

1. Multi-block colors:
   - There can be multiple blocks of each color (e.g. 2 red, 3 yellow, 2 green, 1 blue).
   - For phrases like "the red block" or "a red block", choose a specific one using:
       * nearest: system_state.get_nearest_block_id_by_color("red")
       * farthest: system_state.get_farthest_block_id_by_color("red")
   - Phrases like "the farthest red block" → use get_farthest_block_id_by_color("red").
   - Phrases like "the nearest green block" → use get_nearest_block_id_by_color("green").

2. Stacks:
   - A *stack* is a location with height >= 2 blocks.
   - "the stack" (without qualifiers) → the NEAREST stack:
       (x, y, h) = system_state.get_nearest_stack_xy()
   - "the tallest stack" or "the highest stack" → use:
       (x, y, h) = system_state.get_tallest_stack_xy()
   - "over the stack", "on top of the stack" → place on TOP of that chosen stack:
       z_offset = h * 17.9

3. Picking blocks (including stacked ones):
   - ALWAYS call system_state.update_gripper(block_id) after dobot.pick_block(...).
   - Use stack_level to pick at the correct height:
       block = system_state.get_block(block_id)
       level = block['stack_level']
       pick_z = level * 17.9
       dobot.pick_block(block['pixel_x'], block['pixel_y'], z_offset=pick_z)
       system_state.update_gripper(block_id)
   - This works for blocks on the table (level = 0) and stacked blocks (level > 0).

4. Placing blocks:
   - Before placing at (x, y), compute:
       stack_height = system_state.get_stack_height(x, y)
       z_offset = stack_height * 17.9
       blocks_below = system_state.get_stack_at_location(x, y)
   - Then:
       dobot.place_block(x, y, z_offset)
       system_state.update_block_position(block_id, x, y, z_offset, blocks_below=blocks_below)
       system_state.update_gripper(None)

5. "Next to" placement:
   - When asked to place a block "next to" another block:
       nx, ny = system_state.compute_adjacent_to_block(target_id, direction="right", clearance_mm=15.0)
   - When asked to place a block "next to the stack":
       - If "nearest stack" is implied (just "the stack"):
           nx, ny = system_state.compute_adjacent_to_stack(direction="right", strategy="nearest", clearance_mm=15.0)
       - If "tallest stack" is specified:
           nx, ny = system_state.compute_adjacent_to_stack(direction="right", strategy="tallest", clearance_mm=15.0)
   - Then place using logic from rule 4.

6. Color-to-color stacking (e.g. "place the farthest red block on the nearest green block"):
   - Example pattern:
       red_id = system_state.get_farthest_block_id_by_color("red")
       green_id = system_state.get_nearest_block_id_by_color("green")
       red = system_state.get_block(red_id)
       green = system_state.get_block(green_id)
       # pick red
       pick_z = red['stack_level'] * 17.9
       dobot.pick_block(red['pixel_x'], red['pixel_y'], z_offset=pick_z)
       system_state.update_gripper(red_id)
       time.sleep(0.3)
       # place on top of stack where green lives
       target_x = green['pixel_x']
       target_y = green['pixel_y']
       h = system_state.get_stack_height(target_x, target_y)
       z = h * 17.9
       dobot.place_block(target_x, target_y, z)
       below = system_state.get_stack_at_location(target_x, target_y)
       system_state.update_block_position(red_id, target_x, target_y, z, blocks_below=below)
       system_state.update_gripper(None)

7. Contextual references:
   - "there" or "same place" → use system_state.last_place_location (x,y,z).
   - "that", "it", or "the previous block" → use system_state.get_last_manipulated_block_id().

8. Closest/Farthest (no color):
   - "nearest block" → system_state.get_nearest_block_id()
   - "farthest block" → system_state.get_farthest_block_id()

9. Smart Return Handling:
   - If holding a different block and asked to pick a new one, FIRST place the held block back at its current position
     (using its current pixel_x, pixel_y and stack height), then pick the new one.

10. IMPORTANT EXECUTION STYLE:
   - This environment is stateful across commands. Always rely on:
       * stack_level, in_gripper, stack_map, last_place_location, etc.,
     instead of hardcoding assumptions.

EXAMPLE – Command: "pick the farthest red block and place it on the nearest green block"

CODE:
red_id = system_state.get_farthest_block_id_by_color("red")
green_id = system_state.get_nearest_block_id_by_color("green")
red = system_state.get_block(red_id)
green = system_state.get_block(green_id)
# pick farthest red
pick_z = red['stack_level'] * 17.9
dobot.pick_block(red['pixel_x'], red['pixel_y'], z_offset=pick_z)
system_state.update_gripper(red_id)
time.sleep(0.3)
# place on stack at green location
target_x = green['pixel_x']
target_y = green['pixel_y']
h = system_state.get_stack_height(target_x, target_y)
z = h * 17.9
dobot.place_block(target_x, target_y, z)
below = system_state.get_stack_at_location(target_x, target_y)
system_state.update_block_position(red_id, target_x, target_y, z, blocks_below=below)
system_state.update_gripper(None)

EXAMPLE – Command: "place the blue block right next to the stack"
(choose nearest stack, put blue on table next to it)

CODE:
blue_id = system_state.get_nearest_block_id_by_color("blue")
blue = system_state.get_block(blue_id)
pick_z = blue['stack_level'] * 17.9
dobot.pick_block(blue['pixel_x'], blue['pixel_y'], z_offset=pick_z)
system_state.update_gripper(blue_id)
time.sleep(0.3)
nx, ny = system_state.compute_adjacent_to_stack(direction="right", strategy="nearest", clearance_mm=15.0)
h = system_state.get_stack_height(nx, ny)
z = h * 17.9
dobot.place_block(nx, ny, z)
below = system_state.get_stack_at_location(nx, ny)
system_state.update_block_position(blue_id, nx, ny, z, blocks_below=below)
system_state.update_gripper(None)

EXAMPLE – Command: "place the yellow block over the stack"
(assume tallest stack)

CODE:
yellow_id = system_state.get_nearest_block_id_by_color("yellow")
yellow = system_state.get_block(yellow_id)
pick_z = yellow['stack_level'] * 17.9
dobot.pick_block(yellow['pixel_x'], yellow['pixel_y'], z_offset=pick_z)
system_state.update_gripper(yellow_id)
time.sleep(0.3)
sx, sy, h = system_state.get_tallest_stack_xy()
z = h * 17.9
dobot.place_block(sx, sy, z)
below = system_state.get_stack_at_location(sx, sy)
system_state.update_block_position(yellow_id, sx, sy, z, blocks_below=below)
system_state.update_gripper(None)

CRITICAL: Always use stack_level when picking, and stack_height when placing.

Generate executable Python code for:
"""

    def generate_code(self, user_prompt: str, system_state: SystemState) -> Optional[str]:
        """
        Generate Python code from a natural-language user prompt plus the
        current SystemState.

        The caller should subsequently pass the result into CodeExecutor
        for validation and execution.
        """
        if not self.enabled or self.llm is None:
            print("[LLM] Not available (missing API key).")
            return None

        state_summary = system_state.get_state_summary()
        system_prompt = self.system_prompt_template.format(state_summary=state_summary)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        try:
            print(f"\n[LLM] Analyzing: '{user_prompt}'")
            print(f"[LLM] Current gripper state: {system_state.gripper_status}")
            response = self.llm.invoke(messages)
            generated_code = response.content.strip()

            # Strip markdown fences if the model ignores instructions
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0].strip()
            elif "```" in generated_code:
                generated_code = generated_code.split("```")[1].split("```")[0].strip()

            print(f"[LLM] ✓ Code generated ({len(generated_code)} chars)")
            return generated_code
        except Exception as e:
            print(f"[LLM] ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None


# ---------------------------------------------------------------------------
# Code validator & executor
# ---------------------------------------------------------------------------

class CodeExecutor:
    """
    Validates and safely executes LLM-generated code.

    Safety constraints:
      - No imports / function / class definitions / lambdas.
      - No while-loops (to avoid unbounded execution).
      - Only a small whitelist of method calls on:
            dobot, system_state, time.sleep
      - Only a limited set of bare-name calls (print, len, range, int,
        float, str, round).
    """

    def __init__(self) -> None:
        # Full dotted names for attribute calls
        self.allowed_calls = {
            'dobot.pick_block',
            'dobot.place_block',
            'system_state.update_gripper',
            'system_state.update_block_position',
            'system_state.get_block',
            'system_state.get_stack_at_location',
            'system_state.get_stack_height',
            'system_state.is_gripper_empty',
            'system_state.get_holding_block_id',
            'system_state.get_last_manipulated_block_id',
            'system_state.get_nearest_block_id_by_color',
            'system_state.get_farthest_block_id_by_color',
            'system_state.get_nearest_block_id',
            'system_state.get_farthest_block_id',
            'system_state.get_tallest_stack_xy',
            'system_state.get_nearest_stack_xy',
            'system_state.compute_adjacent_position',
            'system_state.compute_adjacent_to_block',
            'system_state.compute_adjacent_to_stack',
            'time.sleep',
        }
        # Bare name calls allowed (Python builtins we expose in __builtins__)
        self.allowed_bare_calls = {'print', 'len', 'range', 'int', 'float', 'str', 'round'}
        self.max_code_length = 4000

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """
        Return the dotted name for a call if it's an attribute on a simple
        Name (e.g. system_state.get_block) or a bare Name (e.g. print).
        Otherwise return None so validation can fail the code.
        """
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            base = func.value.id
            attr = func.attr
            return f"{base}.{attr}"
        elif isinstance(func, ast.Name):
            return func.id  # bare name call
        return None

    def validate_code(self, code_string: str) -> bool:
        """
        Validate code against a whitelist of structures and allowed calls.

        This is a structural sanity check, not a formal proof of safety.
        """
        if len(code_string) > self.max_code_length:
            print(f"[Validator] ✗ Code too long ({len(code_string)} chars > {self.max_code_length})")
            return False

        try:
            tree = ast.parse(code_string, mode='exec')
        except SyntaxError as e:
            print(f"[Validator] ✗ Syntax error: {e}")
            return False

        try:
            for node in ast.walk(tree):
                # Block imports
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    print("[Validator] ✗ Imports are not allowed")
                    return False

                # Block user-defined functions, classes, lambdas
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    print("[Validator] ✗ Defining functions/classes/lambdas is not allowed")
                    return False

                # Block while loops (safety)
                if isinstance(node, ast.While):
                    print("[Validator] ✗ while-loops are not allowed")
                    return False

                if isinstance(node, ast.Call):
                    name = self._get_call_name(node)
                    if name is None:
                        print("[Validator] ✗ Blocked complex call (not a simple attribute or name)")
                        return False

                    # Attribute call like system_state.get_block
                    if '.' in name:
                        if name not in self.allowed_calls:
                            print(f"[Validator] ✗ Blocked call: {name}")
                            return False
                    else:
                        # Bare-name call: must be in allowed_bare_calls
                        if name not in self.allowed_bare_calls:
                            print(f"[Validator] ✗ Blocked bare call: {name}")
                            return False

            print("[Validator] ✓ Code validated")
            return True
        except Exception as e:
            print(f"[Validator] ✗ Validation error: {e}")
            return False

    def execute_code(self, code_string: str, dobot: DobotController, system_state: SystemState) -> bool:
        """
        Execute validated code in a controlled environment.

        The global namespace includes:
          - dobot
          - system_state
          - time
          - a trimmed __builtins__ with only a few safe functions.
        """
        if not self.validate_code(code_string):
            print("[Executor] Code validation failed, NOT executing")
            return False

        safe_globals = {
            'dobot': dobot,
            'system_state': system_state,
            'time': time,
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'int': int,
                'float': float,
                'str': str,
                'round': round,
            }
        }
        try:
            print("\n[Executor] Executing code...")
            print("-" * 70)
            exec(code_string, safe_globals)
            print("-" * 70)
            print("[Executor] ✓ Execution complete")
            return True
        except Exception as e:
            print(f"\n[Executor] ✗ EXECUTION ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class PickPlaceOrchestrator:
    """Main system coordinator with persistent memory."""

    def __init__(self, port: str = DOBOT_PORT, camera_index: int = CAMERA_INDEX) -> None:
        coord_converter = CoordinateConverter()
        self.dobot = DobotController(port=port, coord_converter=coord_converter)
        self.detector = BlockDetector(camera_index=camera_index)
        self.system_state = SystemState(coord_converter=coord_converter)
        self.llm = LLMCodeGenerator(GROQ_API_KEY)
        self.executor = CodeExecutor()

    def auto_update_blocks(self) -> None:
        """Auto-capture and update block positions (XY only, keep known stack heights)."""
        print("\n" + "=" * 70)
        print("🔄 AUTO-UPDATING BLOCK POSITIONS...")
        print("=" * 70)

        self.dobot.go_home()
        time.sleep(0.5)

        captured_blocks = self.detector.auto_capture_blocks(display_duration=2.0)

        if captured_blocks and len(captured_blocks) > 0:
            print(f"\n[Camera] Detected {len(captured_blocks)} blocks")
            for block in captured_blocks:
                block_id = block['global_id']
                if block_id in self.system_state.blocks:
                    # update only XY; keep z/stack_level/in_gripper as they represent logical state
                    self.system_state.blocks[block_id]['pixel_x'] = block['pixel_x']
                    self.system_state.blocks[block_id]['pixel_y'] = block['pixel_y']
                    self.system_state.blocks[block_id]['current_position']['x'] = block['pixel_x']
                    self.system_state.blocks[block_id]['current_position']['y'] = block['pixel_y']
                else:
                    # New block discovered mid-session
                    self.system_state.blocks[block_id] = {
                        'color': block['color'],
                        'id': block['id'],
                        'pixel_x': block['pixel_x'],
                        'pixel_y': block['pixel_y'],
                        'original_x': block['pixel_x'],
                        'original_y': block['pixel_y'],
                        'current_position': {'x': block['pixel_x'], 'y': block['pixel_y'], 'z': 0.0},
                        'stack_level': 0,
                        'blocks_below': [],
                        'in_gripper': False
                    }

            self.system_state.rebuild_stacks()
            self.system_state.save_state()
            print("\n[System] ✓ Block positions updated and saved")
        else:
            print("\n[Camera] ⚠️ No blocks detected (keeping previous positions)")

    def initialize(self) -> None:
        """Initialize all components with state loading."""
        print("\n" + "=" * 70)
        print(" ENHANCED SYSTEM INITIALIZATION v4.3")
        print("=" * 70)

        # Try to load previous state
        print("\n[STEP 0/3] LOADING PREVIOUS STATE")
        print("-" * 70)
        use_saved_state = False
        if self.system_state.load_state():
            print("\n✓ Previous state loaded successfully!")
            print(f"   - {len(self.system_state.blocks)} blocks remembered")
            print(f"   - {len(self.system_state.conversation_history)} commands in history")

            try:
                answer = input("\nUse saved state? (y/n): ").strip().lower()
            except EOFError:
                answer = 'y'
            use_saved_state = (answer == 'y')

        # Dobot connection
        print("\n[STEP 1/3] DOBOT CONNECTION")
        print("-" * 70)
        if not self.dobot.connect():
            print("\n✗ FATAL: Dobot connection failed")
            sys.exit(1)

        # Home
        print("\n[STEP 2/3] POSITIONING")
        print("-" * 70)
        self.dobot.go_home()
        time.sleep(1)

        if use_saved_state and self.system_state.blocks:
            print("\n[STEP 3/3] BLOCK DETECTION (SKIPPED - using saved state)")
        else:
            print("\n[STEP 3/3] BLOCK DETECTION")
            print("-" * 70)
            detected_blocks = self.detector.detect_all_blocks_live()

            if not detected_blocks:
                print("\n✗ FATAL: No blocks detected")
                self.dobot.close()
                sys.exit(1)

            self.system_state.initialize_blocks(detected_blocks)

        print("\n" + "=" * 70)
        print("✓ INITIALIZATION COMPLETE")
        print("=" * 70)

    def run_interactive_loop(self) -> None:
        """Main interactive command loop."""
        print("\n" + "=" * 70)
        print(" READY FOR COMMANDS - MODE v4.3")
        print("=" * 70)
        print("\nYou can give commands like:")
        print("  - 'pick the farthest red block and place it on the nearest green block'")
        print("  - 'place the blue block right next to the stack'")
        print("  - 'place the yellow block over the stack'")
        print("\nSpecial commands:")
        print("  - 'status'      - Show current state")
        print("  - 'history'     - Show command history")
        print("  - 'stacks'      - Show all stacks")
        print("  - 'home'        - Return to home")
        print("  - 'open camera' - Manual camera view / refresh")
        print("  - 'reset state' - Clear saved state on disk")
        print("  - 'quit'        - Exit")
        print("=" * 70 + "\n")

        while True:
            try:
                user_prompt = input("\n[You] Enter command: ").strip()
            except EOFError:
                user_prompt = "quit"

            if not user_prompt:
                continue

            low = user_prompt.lower()

            if low in ['quit', 'exit', 'q']:
                print("\n[System] Shutting down...")
                break
            elif low == 'status':
                print("\n" + "=" * 70)
                print(self.system_state.get_state_summary())
                print("=" * 70)
                continue
            elif low == 'history':
                print("\n" + "=" * 70)
                print("COMMAND HISTORY:")
                print("-" * 70)
                for i, entry in enumerate(self.system_state.conversation_history, 1):
                    print(f"{i}. '{entry['prompt']}'")
                    print(f"   → {entry['action']} at {entry['timestamp']}")
                    if entry.get('details'):
                        print(f"   Details: {entry['details']}")
                print("=" * 70)
                continue
            elif low == 'stacks':
                print("\n" + "=" * 70)
                print("CURRENT STACKS:")
                print("-" * 70)
                stacks = self.system_state.get_stack_locations(min_height=2)
                if stacks:
                    for s in stacks:
                        loc = s['loc_key']
                        meta = self.system_state.stack_meta.get(loc, {})
                        stack_blocks = self.system_state.stack_map.get(loc, [])
                        colors = [self.system_state.blocks[bid]['color'] for bid in stack_blocks if bid in self.system_state.blocks]
                        print(f"  loc {loc} @ ({s['x']},{s['y']}): height={s['height']} → {stack_blocks} colors={colors}")
                else:
                    print("  No stacks (height>=2) detected")
                print("=" * 70)
                continue
            elif low == 'home':
                self.dobot.go_home()
                continue
            elif low == 'open camera':
                self.dobot.go_home()
                time.sleep(0.5)
                captured_blocks = self.detector.detect_all_blocks_live()
                if captured_blocks:
                    for block in captured_blocks:
                        block_id = block['global_id']
                        if block_id in self.system_state.blocks:
                            self.system_state.blocks[block_id]['pixel_x'] = block['pixel_x']
                            self.system_state.blocks[block_id]['pixel_y'] = block['pixel_y']
                            self.system_state.blocks[block_id]['current_position']['x'] = block['pixel_x']
                            self.system_state.blocks[block_id]['current_position']['y'] = block['pixel_y']
                        else:
                            self.system_state.blocks[block_id] = {
                                'color': block['color'],
                                'id': block['id'],
                                'pixel_x': block['pixel_x'],
                                'pixel_y': block['pixel_y'],
                                'original_x': block['pixel_x'],
                                'original_y': block['pixel_y'],
                                'current_position': {'x': block['pixel_x'], 'y': block['pixel_y'], 'z': 0.0},
                                'stack_level': 0,
                                'blocks_below': [],
                                'in_gripper': False
                            }
                    self.system_state.rebuild_stacks()
                    self.system_state.save_state()
                    print(f"\n✓ Updated {len(captured_blocks)} blocks")
                continue
            elif low == 'reset state':
                try:
                    confirm = input("Are you sure? This will delete saved state (y/n): ").strip().lower()
                except EOFError:
                    confirm = 'n'
                if confirm == 'y':
                    if STATE_FILE.exists():
                        STATE_FILE.unlink()
                        print("[System] ✓ State file deleted")
                    self.system_state = SystemState(coord_converter=self.system_state.coord_converter)
                    print("[System] ✓ State reset - please re-detect blocks on next run")
                continue

            # Natural-language → code via LLM
            generated_code = self.llm.generate_code(user_prompt, self.system_state)
            if generated_code is None:
                print("[System] Failed to generate code (LLM unavailable or error).")
                continue

            print("\n" + "=" * 70)
            print("GENERATED CODE:")
            print("-" * 70)
            print(generated_code)
            print("=" * 70)

            # Confirm execution
            try:
                confirm = input("\nExecute this code? (y/n/show): ").strip().lower()
            except EOFError:
                confirm = 'n'

            if confirm == 'show':
                print("\n" + generated_code)
                try:
                    confirm = input("\nExecute? (y/n): ").strip().lower()
                except EOFError:
                    confirm = 'n'

            if confirm != 'y':
                print("[System] Execution cancelled")
                continue

            success = self.executor.execute_code(generated_code, self.dobot, self.system_state)
            if success:
                self.system_state.add_to_history(
                    user_prompt,
                    "Success",
                    details={'code_length': len(generated_code)}
                )
                print("\n✓ Command completed successfully")
                # Optional: update from camera so positions stay fresh
                self.auto_update_blocks()
            else:
                self.system_state.add_to_history(user_prompt, "Failed")
                print("\n✗ Command failed")
                self.dobot.go_home()

        # Cleanup
        print("\n[System] Cleanup...")
        try:
            self.dobot.go_home()
        except Exception:
            pass
        self.dobot.close()
        print(f"\n✓ System shutdown complete")
        print(f"✓ State saved to {STATE_FILE}")
        print("=" * 70)

    def run(self) -> None:
        """Main entry point for the orchestrator."""
        try:
            self.initialize()
            self.run_interactive_loop()
        except Exception as e:
            print(f"\n✗ FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.dobot.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-powered Dobot pick & place system")
    parser.add_argument(
        "--port",
        default=DOBOT_PORT,
        help=f"Dobot serial port (default: {DOBOT_PORT})"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=CAMERA_INDEX,
        help=f"Camera index (default: {CAMERA_INDEX})"
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip initial 'Establish CONNECTION?' confirmation"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\n📦 LLM Dobot System v{__version__}")
    print(f"📁 State File: {STATE_FILE}")
    if STATE_FILE.exists():
        print("✓ Found existing state file (will offer to load)")
    else:
        print("ℹ No saved state (will create new)")

    print("\n⚙️  Configuration:")
    print(f"   Dobot port:     {args.port}")
    print(f"   Camera index:   {args.camera_index}")
    print(f"   Frame:          {FRAME_WIDTH}x{FRAME_HEIGHT}px")

    if args.yes:
        proceed = 'y'
    else:
        try:
            proceed = input("\n▶ Establish CONNECTION? (y/n): ").strip().lower()
        except EOFError:
            proceed = 'n'

    if proceed == 'y':
        orchestrator = PickPlaceOrchestrator(port=args.port, camera_index=args.camera_index)
        orchestrator.run()
    else:
        print("\nExiting.")


if __name__ == "__main__":
    main()
