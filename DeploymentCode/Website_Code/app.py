import streamlit as st
import pandas as pd
import plotly.express as px
import boto3
import requests
import json
import os
import time  # Imported correctly for sleep()
from datetime import datetime, time as dt_time  # Renamed to avoid conflict
from botocore.exceptions import ClientError
from decimal import Decimal

# --- CONFIGURATION ---
AWS_REGION = "ap-south-1"
DYNAMODB_TABLE = os.environ.get("ISSUES_TABLE_NAME", "IssuesTrackingTable")
INSTALLATIONS_TABLE = os.environ.get("INSTALLATIONS_TABLE_NAME", "github-installations")
EXPERTISE_TABLE = os.environ.get("EXPERTISE_TABLE_NAME", "RepoExpertise")
USER_AVAILABILITY_TABLE = os.environ.get("USER_AVAILABILITY_TABLE_NAME", "UserAvailability")
SECRET_NAME = "dashboard-oauth-credentials"
REDIRECT_URI = "http://65.0.75.51:8501"  

# --- AWS SETUP ---
@st.cache_resource
def get_aws_session():
    return boto3.Session(region_name=AWS_REGION)

def get_oauth_secrets():
    session = get_aws_session()
    client = session.client(service_name='secretsmanager')
    try:
        get_secret_value_response = client.get_secret_value(SecretId=SECRET_NAME)
        secrets = json.loads(get_secret_value_response['SecretString'])
        return secrets['GITHUB_CLIENT_ID'], secrets['GITHUB_CLIENT_SECRET']
    except ClientError as e:
        st.error(f"Failed to load secrets: {e}")
        st.stop()

# --- GITHUB OAUTH FLOW ---
def exchange_code_for_token(client_id, client_secret, code):
    url = "https://github.com/login/oauth/access_token"
    headers = {"Accept": "application/json"}
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": REDIRECT_URI
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json().get("access_token")

def get_github_user_info(token):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    user_resp = requests.get("https://api.github.com/user", headers=headers)
    if user_resp.status_code != 200: return None, []
    user_data = user_resp.json()
    username = user_data['login']
    repos_resp = requests.get("https://api.github.com/user/repos?per_page=100&type=all", headers=headers)
    repos = []
    if repos_resp.status_code == 200:
        all_repos = repos_resp.json()
        repos = [r['full_name'] for r in all_repos if r['permissions']['push']]
    return username, repos

# --- DYNAMODB FUNCTIONS ---

def fetch_installed_repos():
    session = get_aws_session()
    dynamodb = session.resource('dynamodb')
    table = dynamodb.Table(INSTALLATIONS_TABLE)
    try:
        response = table.scan(ProjectionExpression="repo_name")
        items = response.get('Items', [])
        while 'LastEvaluatedKey' in response:
            response = table.scan(ProjectionExpression="repo_name", ExclusiveStartKey=response['LastEvaluatedKey'])
            items.extend(response.get('Items', []))
        return {item['repo_name'] for item in items}
    except Exception as e:
        st.error(f"Error fetching installed repos: {e}")
        return set()

def fetch_repo_issues(repo_name):
    session = get_aws_session()
    dynamodb = session.resource('dynamodb')
    table = dynamodb.Table(DYNAMODB_TABLE)
    try:
        response = table.query(KeyConditionExpression=boto3.dynamodb.conditions.Key('repo_name').eq(repo_name))
        items = response.get('Items', [])
        if not items: return pd.DataFrame()
        df = pd.DataFrame(items)
        
        expected = ['issue_id', 'title', 'issue_type', 'priority', 'status', 'current_assignee', 'created_at', 'component', 'is_spam']
        for c in expected:
            if c not in df.columns: df[c] = None

        df['is_spam_bool'] = df['is_spam'].astype(str).str.lower() == 'spam'
        
        def clean_row(row):
            if row['is_spam_bool']:
                row['issue_type'] = 'Spam'
                row['priority'] = '-'
                row['component'] = '-'
                row['current_assignee'] = '-'
                row['status'] = 'closed (spam)'
            else:
                if pd.isna(row['issue_type']): row['issue_type'] = 'Unclassified'
                if pd.isna(row['priority']): row['priority'] = 'None'
                if pd.isna(row['component']): row['component'] = 'N/A'
                if pd.isna(row['current_assignee']): row['current_assignee'] = 'Unassigned'
            return row

        df = df.apply(clean_row, axis=1)
        df['assigned_to'] = df['current_assignee']
        df['label'] = df['issue_type']
        
        date_col = 'created_at' if 'created_at' in df.columns else 'last_updated_pipeline'
        df['created_at'] = pd.to_datetime(df[date_col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# --- TEAM MANAGEMENT FUNCTIONS ---

def parse_time_str(t_str):
    try:
        return datetime.strptime(str(t_str), "%H:%M").time()
    except (ValueError, TypeError):
        return datetime.strptime("09:00", "%H:%M").time()

def fetch_team_expertise(repo_name):
    session = get_aws_session()
    dynamodb = session.resource('dynamodb')
    exp_table = dynamodb.Table(EXPERTISE_TABLE)
    ua_table = dynamodb.Table(USER_AVAILABILITY_TABLE)
    
    try:
        # 1. Fetch Expertise Profiles
        response = exp_table.get_item(Key={'repo_name': repo_name})
        if 'Item' not in response:
            return pd.DataFrame()
        
        profiles = response['Item'].get('expertise_profiles', {})
        
        # 2. Fetch Availability
        users = list(profiles.keys())
        availability_map = {}
        
        if users:
            for i in range(0, len(users), 100):
                batch_keys = [{'user_handle': u} for u in users[i:i+100]]
                ua_resp = dynamodb.batch_get_item(
                    RequestItems={USER_AVAILABILITY_TABLE: {'Keys': batch_keys}}
                )
                for item in ua_resp.get('Responses', {}).get(USER_AVAILABILITY_TABLE, []):
                    availability_map[item['user_handle']] = item

        flat_data = []
        for username, details in profiles.items():
            skills = details.get('technical_skills', [])
            skills_str = ", ".join(skills) if isinstance(skills, list) else str(skills)
            
            activity = details.get('activity_summary', {})
            total_contribs = int(activity.get('total_contributions', 0))
            
            user_avail = availability_map.get(username, {})
            start_str = user_avail.get('inferred_start_time_utc', '09:00')
            end_str = user_avail.get('inferred_end_time_utc', '17:00')

            flat_data.append({
                "User": username,
                "Role": details.get('inferred_role', 'Contributor'),
                "Skills": skills_str,
                "Start (UTC)": parse_time_str(start_str),
                "End (UTC)": parse_time_str(end_str),
                "Contributions": total_contribs,
                "Confidence": details.get('confidence', 'Low')
            })
            
        return pd.DataFrame(flat_data)

    except Exception as e:
        st.error(f"Error fetching team data: {e}")
        return pd.DataFrame()

def save_team_changes(repo_name, edited_df):
    session = get_aws_session()
    dynamodb = session.resource('dynamodb')
    exp_table = dynamodb.Table(EXPERTISE_TABLE)
    ua_table = dynamodb.Table(USER_AVAILABILITY_TABLE)
    
    try:
        response = exp_table.get_item(Key={'repo_name': repo_name})
        current_item = response.get('Item', {})
        current_profiles = current_item.get('expertise_profiles', {})

        if not current_profiles:
            current_profiles = {}

        for index, row in edited_df.iterrows():
            username = row['User']
            if username in current_profiles:
                current_profiles[username]['inferred_role'] = row['Role']
                skills_list = [s.strip() for s in row['Skills'].split(',') if s.strip()]
                current_profiles[username]['technical_skills'] = skills_list
            
            # Use dt_time for type check
            s_str = row['Start (UTC)'].strftime("%H:%M") if isinstance(row['Start (UTC)'], (datetime, dt_time)) else str(row['Start (UTC)'])
            e_str = row['End (UTC)'].strftime("%H:%M") if isinstance(row['End (UTC)'], (datetime, dt_time)) else str(row['End (UTC)'])
            
            ua_table.put_item(
                Item={
                    'user_handle': username,
                    'inferred_start_time_utc': s_str,
                    'inferred_end_time_utc': e_str,
                    'last_updated': datetime.now().isoformat()
                }
            )
        
        exp_table.update_item(
            Key={'repo_name': repo_name},
            UpdateExpression="SET expertise_profiles = :p",
            ExpressionAttributeValues={':p': current_profiles}
        )
        return True
    except Exception as e:
        st.error(f"Failed to save changes: {e}")
        return False

def add_new_member(repo_name, username, role, skills_str, start_time, end_time):
    session = get_aws_session()
    dynamodb = session.resource('dynamodb')
    exp_table = dynamodb.Table(EXPERTISE_TABLE)
    ua_table = dynamodb.Table(USER_AVAILABILITY_TABLE)
    
    username = username.strip()
    if not username: return False, "Username cannot be empty"

    try:
        response = exp_table.get_item(Key={'repo_name': repo_name})
        current_profiles = response.get('Item', {}).get('expertise_profiles', {})
        
        if username in current_profiles:
            return False, f"User '{username}' already exists."

        new_profile = {
            "username": username,
            "inferred_role": role,
            "technical_skills": [s.strip() for s in skills_str.split(',') if s.strip()],
            "confidence": "Manual",
            "contribution_types": ["Manual Entry"],
            "profile_summary": {"name": username, "bio": "Added via Dashboard"},
            "activity_summary": {"total_contributions": 0, "commits_count": 0, "issues_closed": 0, "prs_merged": 0},
            "repo_context": {}
        }
        current_profiles[username] = new_profile
        
        exp_table.update_item(
            Key={'repo_name': repo_name},
            UpdateExpression="SET expertise_profiles = :p",
            ExpressionAttributeValues={':p': current_profiles}
        )
        
        # Use dt_time for type check
        s_str = start_time.strftime("%H:%M") if isinstance(start_time, (datetime, dt_time)) else str(start_time)
        e_str = end_time.strftime("%H:%M") if isinstance(end_time, (datetime, dt_time)) else str(end_time)

        ua_table.put_item(
            Item={
                'user_handle': username,
                'inferred_start_time_utc': s_str,
                'inferred_end_time_utc': e_str,
                'last_updated': datetime.now().isoformat()
            }
        )
        
        return True, f"Successfully added {username}!"

    except Exception as e:
        return False, f"Error adding member: {e}"

def delete_team_member(repo_name, username):
    session = get_aws_session()
    dynamodb = session.resource('dynamodb')
    exp_table = dynamodb.Table(EXPERTISE_TABLE)
    ua_table = dynamodb.Table(USER_AVAILABILITY_TABLE)

    try:
        # 1. Remove from Expertise
        response = exp_table.get_item(Key={'repo_name': repo_name})
        if 'Item' in response:
            current_profiles = response['Item'].get('expertise_profiles', {})
            if username in current_profiles:
                del current_profiles[username]
                exp_table.update_item(
                    Key={'repo_name': repo_name},
                    UpdateExpression="SET expertise_profiles = :p",
                    ExpressionAttributeValues={':p': current_profiles}
                )

        # 2. Remove from Availability
        ua_table.delete_item(Key={'user_handle': username})

        return True, f"Successfully deleted {username}."
    except Exception as e:
        return False, f"Error deleting member: {e}"

def rename_team_member(repo_name, old_username, new_username):
    session = get_aws_session()
    dynamodb = session.resource('dynamodb')
    exp_table = dynamodb.Table(EXPERTISE_TABLE)
    ua_table = dynamodb.Table(USER_AVAILABILITY_TABLE)

    new_username = new_username.strip()
    if not new_username: return False, "New username cannot be empty."

    try:
        # 1. Handle Expertise Table
        response = exp_table.get_item(Key={'repo_name': repo_name})
        if 'Item' not in response: return False, "Repo not found."
        
        current_profiles = response['Item'].get('expertise_profiles', {})
        
        if old_username not in current_profiles:
            return False, f"User {old_username} not found."
        if new_username in current_profiles:
            return False, f"User {new_username} already exists."

        # Copy and update
        profile_data = current_profiles[old_username]
        profile_data['username'] = new_username
        current_profiles[new_username] = profile_data
        del current_profiles[old_username]

        exp_table.update_item(
            Key={'repo_name': repo_name},
            UpdateExpression="SET expertise_profiles = :p",
            ExpressionAttributeValues={':p': current_profiles}
        )

        # 2. Handle Availability Table
        ua_response = ua_table.get_item(Key={'user_handle': old_username})
        if 'Item' in ua_response:
            item = ua_response['Item']
            item['user_handle'] = new_username
            # Save new, Delete old
            ua_table.put_item(Item=item)
            ua_table.delete_item(Key={'user_handle': old_username})

        return True, f"Successfully renamed {old_username} to {new_username}."

    except Exception as e:
        return False, f"Error renaming member: {e}"

# --- TRIGGER LAMBDAS ---
def trigger_manual_scan(repo_name, hours_back):
    session = get_aws_session()
    lambda_client = session.client('lambda')
    payload = {"repo_name": repo_name, "hours_back": int(hours_back), "manual_trigger": True}
    try:
        lambda_client.invoke(
            FunctionName="manual-trigger-single-repo",
            InvocationType='Event',
            Payload=json.dumps(payload)
        )
        return True, f"Scan triggered for last {hours_back} hours!"
    except Exception as e:
        return False, str(e)

def trigger_availability_inference(repo_name):
    session = get_aws_session()
    lambda_client = session.client('lambda')
    payload = {"repo_name": repo_name, "task": "infer_availability"}
    try:
        lambda_client.invoke(
            FunctionName="infer-availability",
            InvocationType='Event',
            Payload=json.dumps(payload)
        )
        return True, "Availability inference triggered!"
    except Exception as e:
        return False, str(e)

# --- MAIN APP UI ---
st.set_page_config(page_title="IssueOps Dashboard", layout="wide")

st.markdown("""
    <style>
        body { background-color: #0d0d0d; }
        .main { background-color: #0d0d0d; color: #00ff99; }
        div[data-testid="stSidebar"] { background-color: #121212; }
        h1, h2, h3, h4 { color: #00ff99 !important; }
        .metric-label { color: #00ff99 !important; }
        .stMetricValue { color: #ffcc00 !important; }
        div[data-testid="stDataFrame"] div[class^="stDataFrame"] { border: 1px solid #333; }
    </style>
""", unsafe_allow_html=True)

if 'access_token' not in st.session_state: st.session_state.access_token = None
if "code" in st.query_params and st.session_state.access_token is None:
    token = exchange_code_for_token(get_oauth_secrets()[0], get_oauth_secrets()[1], st.query_params["code"])
    if token:
        st.session_state.access_token = token
        st.query_params.clear()
        st.rerun()

if st.session_state.access_token is None:
    cid, _ = get_oauth_secrets()
    st.title("IssueOps Login")
    st.write("Please log in with GitHub.")
    st.markdown(f'<a href="https://github.com/login/oauth/authorize?client_id={cid}&scope=repo,user" target="_self"><button style="background-color:#00ff99; color:black; padding:10px 20px; border:none; border-radius:5px; font-weight:bold;">Login with GitHub</button></a>', unsafe_allow_html=True)
    st.stop()

username, github_repos = get_github_user_info(st.session_state.access_token)
installed_repos = fetch_installed_repos()
available_repos = [r for r in github_repos if r in installed_repos]

if not available_repos:
    st.warning(f"Hi @{username}! App is not installed on any of your repos.")
    st.stop()

st.sidebar.title(f"{username}")
selected_repo = st.sidebar.selectbox("Select Repository", available_repos)
if st.sidebar.button("Logout"):
    st.session_state.access_token = None
    st.rerun()

tab1, tab2 = st.tabs(["Issue Dashboard", "Team & Expertise"])

with tab1:
    st.title(f"Dashboard: {selected_repo}")
    if selected_repo:
        data = fetch_repo_issues(selected_repo)
        if data.empty:
            st.info("No issues found. Trigger a scan.")
            if st.button("Trigger First Scan"):
                 with st.spinner("Triggering..."):
                    s, m = trigger_manual_scan(selected_repo, 24)
                    if s: st.success(m)
                    else: st.error(m)
        else:
            spam_df = data[data['is_spam_bool'] == True]
            active_df = data[data['is_spam_bool'] == False]
            
            open_c = len(active_df[active_df['status'] == 'open'])
            closed_c = len(active_df[active_df['status'] == 'closed']) + len(spam_df)
            high_c = len(active_df[active_df['priority'].astype(str).str.lower().isin(['high', 'critical', 'p0', 'p1'])])
            unassigned = len(active_df[active_df['assigned_to'] == 'Unassigned'])

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total", len(data))
            c2.metric("Open", open_c)
            c3.metric("Closed", closed_c)
            c4.metric("High Priority", high_c)
            c5.metric("Unassigned", unassigned)
            
            st.divider()
            
            gc1, gc2 = st.columns(2)
            with gc1:
                st.subheader("Distribution by Type")
                plot_data = data[data['is_spam_bool'] == False]
                if not plot_data.empty and 'label' in plot_data.columns:
                    fig = px.pie(plot_data['label'].value_counts().reset_index(), values='count', names='label', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
                    fig.update_layout(paper_bgcolor="#0d0d0d", font_color="#00ff99")
                    st.plotly_chart(fig, use_container_width=True)
            with gc2:
                st.subheader("Distribution by Priority")
                if not plot_data.empty and 'priority' in plot_data.columns:
                    fig2 = px.bar(plot_data['priority'].value_counts().reset_index(), x='priority', y='count', color='priority', color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig2.update_layout(paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d", font_color="#00ff99")
                    st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Issue Details")
            clean_headers = {'issue_id': 'Issue ID', 'title': 'Title', 'component': 'Component', 'priority': 'Priority', 'label': 'Label', 'assigned_to': 'Assigned To', 'status': 'Status'}
            st.dataframe(data[['issue_id', 'title', 'component', 'priority', 'label', 'assigned_to', 'status']].rename(columns=clean_headers), width='stretch')

        st.divider()
        st.subheader("Operations")
        col_op1, col_op2 = st.columns([1, 2])
        with col_op1:
            hours_back = st.number_input("Lookback (Hours)", min_value=1, value=24)
            
            st.write("") # Optional: Adds a tiny bit of vertical spacing
            
            # Button is now inside col_op1, directly below the input
            if st.button("Trigger Pipeline Now"):
                with st.spinner("Triggering..."):
                    s, m = trigger_manual_scan(selected_repo, hours_back)
                    if s: st.success(m)
                    else: st.error(m)

with tab2:
    st.title(f"Team: {selected_repo}")
    st.caption("Manage team roles, skills, and working hours (UTC).")
    
    with st.expander("Add New Member"):
        with st.form("add_member_form"):
            c1, c2 = st.columns(2)
            with c1:
                nu = st.text_input("GitHub Username")
                nr = st.selectbox("Role", ["Core Maintainer", "Active Contributor", "Reviewer", "Contributor"])
                ns = st.text_input("Skills", placeholder="Python, AWS")
            with c2:
                nst = st.time_input("Start Time (UTC)", value=datetime.strptime("09:00", "%H:%M").time())
                net = st.time_input("End Time (UTC)", value=datetime.strptime("17:00", "%H:%M").time())
            
            if st.form_submit_button("Add Member"):
                s, m = add_new_member(selected_repo, nu, nr, ns, nst, net)
                if s:
                    st.success(m)
                    time.sleep(1)
                    st.rerun()
                else: st.error(m)
    
    st.write("")
    team_df = fetch_team_expertise(selected_repo)
    
    if not team_df.empty:
        if st.button("Auto-Infer Timings"):
             with st.spinner("Launching inference job..."):
                 s, m = trigger_availability_inference(selected_repo)
                 if s: st.success(m)
                 else: st.error(m)

        edited_df = st.data_editor(
            team_df,
            column_config={
                "User": st.column_config.TextColumn(disabled=True),
                "Role": st.column_config.SelectboxColumn("Role", options=["Core Maintainer", "Active Contributor", "Reviewer", "Contributor"], required=True),
                "Skills": st.column_config.TextColumn("Skills (comma-separated)"),
                "Start (UTC)": st.column_config.TimeColumn("Start (UTC)", format="HH:mm"),
                "End (UTC)": st.column_config.TimeColumn("End (UTC)", format="HH:mm"),
                "Contributions": st.column_config.NumberColumn(disabled=True),
                "Confidence": st.column_config.TextColumn(disabled=True),
            },
            use_container_width=True,
            hide_index=True,
            num_rows="fixed"
        )
        
        if st.button("Save Changes"):
            with st.spinner("Updating DynamoDB..."):
                if save_team_changes(selected_repo, edited_df):
                    st.success("✅ Updated successfully!")
                    time.sleep(1)
                    st.rerun()
    else:
        st.info("No team data found.")

    st.divider()
    st.subheader("Danger Zone")
    with st.expander("Manage Members (Rename / Delete)"):
        t1, t2 = st.tabs(["Rename Member", "Delete Member"])
        
        with t1:
            st.write("**Rename a Member** (Migrates skills & history)")
            c_r1, c_r2 = st.columns(2)
            with c_r1: old_name = st.text_input("Current Username")
            with c_r2: new_name = st.text_input("New Username")
            if st.button("Rename Member"):
                if old_name and new_name:
                    with st.spinner("Renaming..."):
                        s, m = rename_team_member(selected_repo, old_name, new_name)
                        if s: st.success(m); time.sleep(1); st.rerun()
                        else: st.error(m)
                else: st.error("Please fill both names.")

        with t2:
            st.write("**Delete a Member** (Cannot be undone)")
            del_name = st.text_input("Username to Delete")
            if st.button("Delete Member", type="primary"):
                if del_name:
                    with st.spinner("Deleting..."):
                        s, m = delete_team_member(selected_repo, del_name)
                        if s: st.success(m); time.sleep(1); st.rerun()
                        else: st.error(m)
                else: st.error("Please enter a username.")