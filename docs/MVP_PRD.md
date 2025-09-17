### The Problem

Manual GitHub issue triage is a major bottleneck in software development. It's a slow, repetitive process that drains developer productivity and leads to burnout. The average developer loses **7.3 hours per week** to triaging, duplicates, and notifications. This can lead to delays of over three days before issues are even categorized or routed. Our goal is to automate this entire workflow.

---

### The Solution: IssueOps MVP

We'll build an AI-powered GitHub Marketplace App called IssueOps that automatically manages issues. Our MVP has a very narrow focus to prove the core concept of classification and labeling. It will include four key agents that handle a complete, end-to-end loop.

---

### MVP Functional Deep Dive

The 2-week MVP is a focused, automated system that reads a newly created GitHub issue and uses a series of agents to process it. The core workflow is an end-to-end loop: a GitHub issue is created, a webhook triggers a serverless function, and the issue is classified and updated via an API.

#### **The Four MVP Agents**

1.  **Intake Agent**: This agent will be the "first gatekeeper". It will perform **duplicate detection** to find and link similar issues.
2.  **Triaging Agent**: This agent will classify the issue. It will use an AI model to determine the issue type, such as a **bug** or a **feature request**.
3.  **Assignment Agent**: This agent will perform a **basic assignment** to the correct developer type. It will map issue labels to a contributor's expertise to ensure the right person gets the job.
4.  **Notification Agent**: This agent will **send a notification** to the relevant team or individual. It will be able to route updates through different channels like Slack or email.

#### **Out of Scope for the MVP** 
To deliver a working prototype quickly, the MVP will intentionally postpone several features. It will not perform functions such as: 
* Spam detection 
* Stale issue identification 
* SLA escalation 
* Providing a personalized digest or an agent dashboard. 

Our 2-week MVP's sole job is to prove the core concept of classification and labeling.

---

### Technology Stack

The MVP's technology stack is designed to be simple and modular.

* **Compute**: A **Serverless Function** (e.g., AWS Lambda) for its low cost and zero server management.
* **AI Model**: A Pretrained zero-shot classification model, which provides an intelligent result with minimal training/fine-tuning.
* **Local Testing**: **ngrok** to bridge GitHub to our local machines for testing webhooks.
* **Core Platform**: A GitHub App running in a **Docker container**, using **Python + Flask** for webhook handling.

---

## Two-Week Action Plan for the MVP

#### **Week 1: Establish the Connection**  
**Goal:** Build the foundation and verify GitHub → App connectivity.  

* Set up Python + Flask server to receive GitHub issue webhooks (intake data).  
* Connect to a test repo using ngrok.  
* Log new issue events to confirm the intake pipeline works.  

#### **Week 2: Add Intelligence & Close the Loop**  
**Goal:** Deliver a working end-to-end demo with all four agents.  

* **Intake Agent:** Detect and link duplicates from incoming issues.  
* **Triaging Agent:** Use zero-shot AI model to classify issues (bug, feature, etc.).  
* **Assignment Agent:** Map labels to contributors for basic assignment.  
* **Notification Agent:** Send updates to GitHub + notify via Slack/email.  
* Package app in Docker and deploy to a serverless platform.  
* Validate the complete workflow with a demo issue.  
