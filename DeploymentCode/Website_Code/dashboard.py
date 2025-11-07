import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# Page + theme
st.set_page_config(page_title="IssueOps Dashboard", layout="wide")
st.markdown(
    """
    <style>
        body { background-color: #0d0d0d; }
        .main { background-color: #0d0d0d; color: #00ff99; }
        div[data-testid="stSidebar"] { background-color: #121212; }
        h1, h2, h3, h4 { color: #00ff99 !important; }
        .metric-label { color: #00ff99 !important; }
        .stMetricValue { color: #ffcc00 !important; }
        .streamlit-expanderHeader { color: #00ff99 !important; }
        .stTable td, .stTable th { color: #c7f9d3 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------
# Sample data (replace with real GitHub API / DB)
# ------------------------
now = pd.Timestamp('2025-11-07')
num = 60
np.random.seed(1)
created = pd.to_datetime(np.random.choice(pd.date_range('2025-06-01', now, freq='D'), num))
closed = [d + pd.Timedelta(days=int(x)) if np.random.rand() > 0.3 else pd.NaT for d,x in zip(created, np.random.exponential(8, num))]
labels = np.random.choice(['bug', 'enhancement', 'spam', 'documentation', 'question'], num)
priorities = np.random.choice(['P0','P1','P2','P3'], num, p=[0.05,0.15,0.4,0.4])
assignees = np.random.choice(['Vinay','Ashwin','Anmol','Sai Harsh','Unassigned','Ranjith','Priya'], num)
reporters = np.random.choice(['alice','bob','carol','dave','eve','frank'], num)
comments = np.random.poisson(2, num)
last_updated = [c + pd.Timedelta(days=int(d)) for c,d in zip(created, np.random.randint(0,50,num))]
status = ['closed' if pd.notnull(c) else 'open' for c in closed]
closed_at = pd.to_datetime(closed)

data = pd.DataFrame({
    'issue_id': np.arange(1000,1000+num),
    'title':[f'Issue {i}' for i in range(num)],
    'label': labels,
    'priority': priorities,
    'assigned_to': assignees,
    'reporter': reporters,
    'comments': comments,
    'created_at': created,
    'last_updated': pd.to_datetime(last_updated),
    'status': status,
    'closed_at': closed_at,
})

# Derived fields
stale_days_threshold = 30
data['days_since_update'] = (now - data['last_updated']).dt.days
data['is_stale'] = (data['status'] == 'open') & (data['days_since_update'] >= stale_days_threshold)

def avg_close_days(df):
    closed = df[df['status']=='closed'].copy()
    if closed.empty: return np.nan
    return (closed['closed_at'] - closed['created_at']).dt.days.mean()

# Sidebar controls
st.sidebar.header('⚙️ Controls')
st.sidebar.markdown('Filter issues and refresh mock data')
label_filter = st.sidebar.multiselect('Labels', sorted(data['label'].unique()), default=list(data['label'].unique()))
priority_filter = st.sidebar.multiselect('Priorities', sorted(data['priority'].unique()), default=list(data['priority'].unique()))
assignee_filter = st.sidebar.multiselect('Assignees', sorted(data['assigned_to'].unique()), default=list(data['assigned_to'].unique()))
days_stale = st.sidebar.slider('Stale threshold (days)', 7, 90, value=stale_days_threshold)
refresh = st.sidebar.button('Refresh (mock)')

if refresh:
    st.experimental_rerun()

# apply filters
filtered = data[data['label'].isin(label_filter) & data['priority'].isin(priority_filter) & data['assigned_to'].isin(assignee_filter)].copy()

# Header
st.markdown("<h1 style='color:#00ff99;'>💹 Metrics Terminal</h1>", unsafe_allow_html=True)
st.caption('From newbie-friendly stats to senior-maintainer KPIs.')

# Top KPIs
col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
col1.metric('Total Issues Created', len(filtered))
col2.metric('Open Issues', len(filtered[filtered['status']=='open']))
col3.metric('Closed Issues', int(filtered['is_stale'].sum()))
col4.metric('Stale Issues', f"{avg_close_days(filtered):.1f}" if not np.isnan(avg_close_days(filtered)) else 'n/a')
col5.metric('Pull Requests', f"{filtered['comments'].mean():.1f}")

st.divider()

# Left column: Trends & distributions
left, right = st.columns((3,3))
with left:
    st.subheader('📈 Trends')
    # Issues over time
    # --- Time Series Chart ---
    time_series = (
        filtered
        .groupby(filtered['created_at'].dt.to_period('W'))
        .size()
        .reset_index(name='Count')
    )
    time_series['Time'] = time_series['created_at'].dt.to_timestamp()

    # you can safely rename the chart title or labels here
    fig_ts = px.line(
        time_series,
        x='Time',
        y='Count',
        labels={'Time': 'Time', 'Count': 'Count'}
    )

    fig_ts.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d0d0d',
        font_color='#00ff99'
    )

    st.plotly_chart(fig_ts, use_container_width=True)



    
with right:
    st.subheader('👥 Contributors & Activity')
    # contributors derived
    contributors = pd.concat([
        filtered[['assigned_to']].rename(columns={'assigned_to': 'User'}),
        filtered[['reporter']].rename(columns={'reporter': 'User'})
    ])

    contrib_stats = contributors['User'].value_counts().reset_index()
    contrib_stats.columns = ['User', 'Activity Count']

    # last activity per user
    last_act = pd.concat([
        filtered[['assigned_to', 'last_updated']].rename(columns={'assigned_to': 'User', 'last_updated': 'Last Updated'}),
        filtered[['reporter', 'last_updated']].rename(columns={'reporter': 'User', 'last_updated': 'Last Updated'})
    ])
    last_act = last_act.groupby('User')['Last Updated'].max().reset_index()

    # merge and compute activity days
    contrib_stats = contrib_stats.merge(last_act, on='User', how='left')
    contrib_stats['Days Since Active'] = (now - contrib_stats['Last Updated']).dt.days

    # format Last Updated date like "7 October 2025"
    contrib_stats['Last Updated'] = contrib_stats['Last Updated'].dt.strftime('%-d %B %Y')

    # display dataframe with correct capitalization and formatting
    st.dataframe(
        contrib_stats.sort_values('Activity Count', ascending=False).reset_index(drop=True),
        use_container_width=True
    )

# ==============================
# 📊 Detailed Pie Charts Section
# ==============================

st.subheader(" Issue Distribution Overview")

col1, col2 = st.columns(2)

# ---- PIE 1: Priority Levels ----
with col1:
    st.markdown("### 🔺 Priority Levels")

    priority_counts = (
        filtered['priority']
        .value_counts()
        .reset_index()
    )
    priority_counts.columns = ['Priority', 'Count']  # enforce correct column names

    fig_priority = px.pie(
        data_frame=priority_counts,
        names='Priority',
        values='Count',
        hole=0.4,
        color='Priority',
        color_discrete_map={
            'P0': '#ff3300',
            'P1': '#ff6600',
            'P2': '#ffaa00',
            'P3': '#ffcc00'
        },
        title='Distribution by Priority Levels'
    )

    fig_priority.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d0d0d',
        font_color='#00ff99',
        legend_title_text='Priority',
        legend=dict(orientation='h', y=-0.2, x=0.25)
    )
    st.plotly_chart(fig_priority, width='stretch')  # updated Streamlit syntax


# ---- PIE 2: Issue Types (Bug / Enhancement / Question) ----
with col2:
    st.markdown("### 🧩 Issue Categories")

    type_counts = (
        filtered[filtered['label'].isin(['bug', 'enhancement', 'question'])]
        ['label']
        .value_counts()
        .reset_index()
    )
    type_counts.columns = ['Type', 'Count']  # enforce correct names

    fig_type = px.pie(
        data_frame=type_counts,
        names='Type',
        values='Count',
        hole=0.4,
        color='Type',
        color_discrete_map={
            'bug': '#ff3333',
            'enhancement': '#00cc99',
            'question': '#3399ff'
        },
        title='Distribution of Bugs, Enhancements & Questions'
    )

    fig_type.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d0d0d',
        font_color='#00ff99',
        legend_title_text='Issue Type',
        legend=dict(orientation='h', y=-0.2, x=0.25)
    )
    st.plotly_chart(fig_type, width='stretch')


# Middle: Detailed lists and tables
st.subheader('🗂️ Issues Table (filtered)')
visible_cols = ['issue_id','title','label','priority','assigned_to','reporter','comments','status','created_at','last_updated','is_stale']
st.dataframe(filtered[visible_cols].sort_values('created_at', ascending=False))

# Stale issues expander
with st.expander('🚨 Stale Issues (open, no activity)'):
    stale_df = filtered[filtered['is_stale']].copy()
    if stale_df.empty:
        st.info('No stale issues in the current filter set.')
    else:
        st.table(stale_df[visible_cols + ['days_since_update']].sort_values('days_since_update', ascending=False))
        st.markdown('**Action ideas:** ping assignees, close-by-stale policy, create triage task.')

# Top reporters and commenters
st.subheader('⭐ Top Reporters & Comment Activity')
reporters = filtered['reporter'].value_counts().reset_index()
reporters.columns = ['reporter','count']
st.table(reporters.head(10))

# Metrics for maintainers
colA, colB, colC = st.columns(3)
colA.metric('Median time to close (days)', f"{filtered[filtered['status']=='closed'].assign(td=(filtered['closed_at']-filtered['created_at']).dt.days)['td'].median():.1f}" if not filtered[filtered['status']=='closed'].empty else 'n/a')
colB.metric('Percent spam', f"{100*len(filtered[filtered['label']=='spam'])/len(filtered):.1f}%")
colC.metric('Dup rate (simulated)', f"{np.random.rand()*10:.1f}%")

# Export and quick actions
st.divider()
st.subheader('🔁 Export & Quick Actions')
csv = filtered.to_csv(index=False)
st.download_button('Download CSV of filtered issues', data=csv, file_name='issues_export.csv', mime='text/csv')

if st.button('Mark selected stale as stale-tag (mock)'):
    st.success('Marked stale items with label "stale" (mock action).')

# Footer note
st.markdown("<hr><center style='color:#00ff99;'>IssueOps Cloud Intelligence Terminal © 2025</center>", unsafe_allow_html=True)
