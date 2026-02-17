🚀 LLM-Powered Vision-Language Robotic Manipulation System

RAS 545 – Robotics & AI Systems | Arizona State University
Under the guidance of Professor Sangram Redkar

I developed an advanced LLM-driven robotic manipulation system that enables a Dobot Magician Lite robotic arm to interpret natural language instructions and autonomously execute multi-step pick-and-place tasks in a dynamic workspace.

🧠 Language-to-Action Intelligence

Integrated a Groq-hosted Llama 3.3 large language model to interpret high-level natural language commands such as:

“Pick the farthest red block.”

“Place the blue block next to the tallest stack.”

Designed a structured prompting framework where the LLM generates validated Python action plans under a strict manipulation rule set.

Implemented AST-based code validation and a controlled execution sandbox to ensure safe and deterministic robot behavior.

Enabled contextual reasoning capabilities, allowing references such as:

“That block”

“Place it where it was before”

Sequential multi-step instructions

👁️ Vision & Workspace Calibration

Built a complete OpenCV-based color detection pipeline for real-time object identification and localization.

Developed an affine calibration model to accurately transform pixel coordinates into Dobot world coordinates (millimeters), enabling precise physical manipulation.

Ensured robustness under varying lighting conditions and object configurations.

🧠 Agentic Memory & World Modeling

Designed a persistent world-state memory system that tracks:

Block identities and positions

Stack heights and spatial relationships

Gripper state (open/closed)

Last manipulated object

Command history

This memory layer enables:

Multi-step task planning

Stack-aware reasoning

Spatial adjacency logic

Context retention across commands

The system behaves as a goal-directed embodied AI agent rather than a simple rule-based controller.

🤖 Robotic Control & Execution

Implemented a height-aware pick-and-place controller with smooth trajectory planning and suction-based gripping.

Designed safety mechanisms including:

Action schema validation

Function whitelisting

Workspace boundary enforcement

Ensured deterministic execution even when LLM outputs are involved.

🔬 Technical Significance

This project demonstrates a hybrid Vision–Language–Action architecture, combining:

Computer vision for perception

LLM reasoning for decision-making

Structured program synthesis

Agentic memory modeling

Safe robotic execution

It reflects modern embodied AI system design principles used in advanced robotics research.
