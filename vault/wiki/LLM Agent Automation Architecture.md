# LLM Agent Automation Architecture

**Summary:**
This meeting discusses the architecture of LLM agent systems, focusing on the concept of "harnesses" – the software that manages and controls LLMs, enabling them to interact with code, access memory (stored as Markdown files), and automate tasks. The discussion highlights the role of artifacts (AI-generated files) as a source of context and the importance of organized memory management for efficient agent operation.

**Key Concepts:**

*   **LLM Agents:** Systems that utilize LLMs to interact with code and perform tasks.
*   **Harness:** The software layer surrounding an LLM that manages its behavior, including accessing memory, using tools, and controlling the overall workflow. It's likened to the "engine" of a car, while the LLM is the "steering wheel" or the "parts" that are being developed.
*   **Artifacts:** AI-generated files (typically Markdown) that serve as the primary form of memory storage for LLM agents. These files are used to retain context and information from previous interactions.
*   **Memory Management:** The process of organizing and retrieving artifacts to provide LLMs with relevant context for generating outputs.
*   **Context:** The information an LLM uses to generate a response, primarily derived from artifacts.

**Source Reference:** [[source: vault/raw/Meeting Title: LLM agents, harnesses.md]]

**Related Notes:**

*   [[LLM Agents]]
*   [[Memory Management in AI]]
*   [[Code Interpreters]]