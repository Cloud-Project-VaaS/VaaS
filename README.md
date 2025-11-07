# GitHub App — IssueOps


## App details

IssueOps is a GitHub App that automates repository triage and routing with zero manual setup from repository owners. Install it, and the app automatically checks your repository every 10 minutes to detect new issues or updates, then applies predefined automation and checks. Key capabilities:

- **Automated triage**
  - Classifies new issues using rule based and ML assisted logic.
  - Applies labels, templates, and priority tags automatically.

- **Routing and Analysis**
  - Matches issues to maintainers or teams using mapping rules.
  - Automatically assigns or notifies the correct owner when an issue is opened.
  - Adds or updates labels, and closes or flags stale or duplicate issues according to policy.
  - Sends aggregated metrics to the project dashboard for instructors or maintainers to review.

- **Security and privacy**
  - Only accesses repositories you explicitly select during installation.
  - Does not require repository owners to add files, change code, or run local scripts.


---

## How to use the app

### 🚀 Installation Steps
1. Click the installation link:  
   👉 **[Install the App](https://github.com/apps/ds252-issueops)**

2. On the GitHub page:
   - Click **Install**.
   - Choose **"Only select repositories"**.
   - Select the repository (or repositories) you want the app to manage.
   - Click **Install** to confirm.

That is all. The app will start working automatically on the selected repository.


### 📊 Dashboard / Metrics
You can view all collected metrics and repository insights on the dashboard. Some metrics are still under development and may not be fully implemented yet:  
👉 **[Visit Dashboard](http://13.127.133.1:5000)**

The dashboard provides a comprehensive view of your repository’s activity, including:

- **📈 Issue Overview** — total issues created, open issues, closed issues, stale issues, and pull request counts.  
- **👥 Contributors & Activity** — active contributors, last activity date, and activity trends.  
- **🧩 Issue Distribution** — visual breakdown of issues by **priority level** and **category** (e.g., bug, enhancement, question).  
- **📅 Trends** — timeline view showing activity frequency over time.  
- **🗂️ Issues Table (Filtered)** — complete table of issues with details such as ID, title, label, priority, assignee, status, and timestamps.

This dashboard updates automatically every few minutes as the app gathers new data from your repository.



### Troubleshooting & contact
If the app does not appear or automation is not running:
- Confirm you installed it on the correct repository.
- Confirm the app has the permissions above.
- If you face any issues please contact our team.

---
