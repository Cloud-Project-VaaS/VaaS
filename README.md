#  IssueOps: Intelligent Repository Management

> **Automate Triage. Optimize Routing. Empower Teams.**

**IssueOps** is a comprehensive GitHub App and Dashboard solution designed to streamline your development workflow. By leveraging Machine Learning and intelligent routing, it automates the tedious parts of issue management, allowing your team to focus on what matters: **writing code**.

---

##  Key Features

### Automated Triage & Classification
- **Smart Component Detection**: Uses LLMs to analyze issue content and classify it into one of 7 distinct components:
  - **Frontend** 
  - **Backend** 
  - **Database** 
  - **DevOps** 
  - **Mobile** 
  - **Documentation** 
  - **Security** 
- **Issue Type Classification**: Automatically categorizes issues into **Bug**, **Enhancement**, or **Question**.
- **Priority Assignment**: Analyzes urgency to assign priority levels (High, Medium, Low).

- **Spam Filtering**: Automatically flags and closes spam or low-quality issues.

### Intelligent Routing
- **Expertise Matching**: Routes issues to the best-suited maintainer based on their past contributions and skills.
- **Availability-Aware**: Checks team availability (working hours) before assigning tasks to prevent burnout.
- **Dynamic Learning**: Continuously updates contributor profiles based on their latest activity.

### Interactive Dashboard
- **Real-time Metrics**: View issue trends, closure rates, and backlog health.
- **Team Management**: Manage roles, skills, and working hours for all contributors.
- **Manual Triggers**: Manually trigger scans and pipeline runs for testing and immediate updates.

---

## Architecture & Tech Stack
This project is built on a modern, 100% serverless, event-driven architecture designed for scalability, security, and low operational overhead.

### **Core Architecture**
- Provider: Amazon Web Services (AWS)
- Pattern: Event-Driven / Asynchronous
- Identity: GitHub App (JWT-based authentication)

### **Tech Stack**
#### Compute & Logic
- AWS Lambda (Python 3.12): Handles all core logic, including issue fetching, classification, assignment, and SLA enforcement.

#### Orchestration & Events
- Amazon EventBridge: Manages the event bus that decouples the pipeline (e.g., issue.batch.new triggers classify_spam).
- Scheduled Rules: Triggers hourly ingestion jobs and daily SLA checks.

#### Data Storage
- Amazon DynamoDB: Serves as the single source of truth for:
- Issues: Tracking lifecycle, labels, and assignment (IssuesTrackingTable).
- Intelligence: Storing inferred expertise and team availability (RepoExpertise, UserAvailability).
- Installations: Managing active GitHub App installations (github-installations).

#### Intelligence & AI
- AWS Bedrock: Provides access to Foundational Models.
- Mistral 7B Instruct (v0:2): Used for inferring developer expertise and team structures.
- Llama 3 8B Instruct: Used for spam filtering and metadata enrichment (rewriting titles/bodies).
- DeepSeek V3: Powering the "Master Agent" for final issue assignment decisions.

#### Security
- AWS Secrets Manager: Securely stores GitHub App private keys and OAuth client secrets.
- IAM Roles: Least-privilege roles assigned to each Lambda function and the EC2 instance.

#### Frontend / Control Plane
- Streamlit: A Python-based dashboard for monitoring, configuration, and manual triggers.
- Amazon EC2 (t3.small): Hosts the dashboard, secured with an IAM Instance Profile.
- GitHub OAuth: Authenticates admin users and authorizes access based on repository permissions.

---

## Getting Started

### 1. Install the GitHub App
1.  Navigate to the **[App Installation Page](https://github.com/apps/ds252-issueops)**.
2.  Click **Install** and select the repositories you want to manage.
3.  *That's it!* The app will start scanning for new issues every hour.

### 2. Access the Dashboard
The dashboard provides deep insights and control over the automation.

- **URL**: `http://65.0.75.51:8501/` 
- **Features**:
    - Visualize issue distribution.
    - Update team expertise profiles.
    - Dedicated button for manual issue scans.

---

## Screenshots
![Dashboard Preview](images/image2.png)

---


### Automated Labeling in Action
The bot automatically analyzes and labels issues directly on GitHub.
![GitHub Bot Action](images/image1.png)



---
