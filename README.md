#  IssueOps: Intelligent Repository Management

> **Automate Triage. Optimize Routing. Empower Teams.**

**IssueOps** is a comprehensive GitHub App and Dashboard solution designed to streamline your development workflow. By leveraging Machine Learning and intelligent routing, it automates the tedious parts of issue management, allowing your team to focus on what matters: **writing code**.

---

##  Key Features

### 🤖 Automated Triage & Classification
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

### 🧠 Intelligent Routing
- **Expertise Matching**: Routes issues to the best-suited maintainer based on their past contributions and skills.
- **Availability-Aware**: Checks team availability (working hours) before assigning tasks to prevent burnout.
- **Dynamic Learning**: Continuously updates contributor profiles based on their latest activity.

### 📊 Interactive Dashboard
- **Real-time Metrics**: View issue trends, closure rates, and backlog health.
- **Team Management**: Manage roles, skills, and working hours for all contributors.
- **Manual Triggers**: Manually trigger scans and pipeline runs for testing and immediate updates.

---

## 🛠️ Architecture & Tech Stack

Built with a modern, serverless-first architecture:

- **Core Logic**: AWS Lambda (Python)
- **Database**: Amazon DynamoDB
- **Frontend/Dashboard**: Streamlit
- **Infrastructure**: AWS (AP-South-1)

---

## 🚀 Getting Started

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

![Dashboard Preview](images/image2.png)

---

## 📸 Screenshots

### Automated Labeling in Action
The bot automatically analyzes and labels issues directly on GitHub.
![GitHub Bot Action](images/image1.png)



---

<p align="center">
  Made with ❤️ by the IssueOps Team
</p>
