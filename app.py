import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px

# 1. Page Layout Configuration
st.set_page_config(page_title="Interactive Stats App", layout="wide", page_icon="📊")

st.title("📊 Interactive Further Statistics Dashboard")
st.write("Upload any CSV dataset to generate interactive charts and live hypothesis tests instantly.")

# 2. Generic File Uploader Widget
uploaded_file = st.file_uploader("Upload your CSV file here:", type="csv")

if uploaded_file is not None:
    # Read the generic user data into a dataframe
    df = pd.read_csv(uploaded_file)
    
    # Setup Layout Tabs
    tab1, tab2 = st.tabs(["📈 Data Explorer & Visuals", "🧮 Live Statistical Testing"])
    
    # ----------------------------------------------------
    # TAB 1: VISUALIZATION & PREVIEW
    # ----------------------------------------------------
    with tab1:
        st.subheader("📋 Quick Metrics & Preview")
        
        # FIX: Explicitly extraction of single integer values from df.shape tuple
        total_rows = int(df.shape[0])
        total_cols = int(df.shape[1])
        missing_vals = int(df.isna().sum().sum())
        
        # Display high-level metrics cards smoothly
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("1. Total Rows (n)", f"{total_rows:,}")
        col_m2.metric("2. Total Columns", f"{total_cols:,}")
        col_m3.metric("3. Missing Values", f"{missing_vals:,}")
        
        # Dataframe preview
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        st.subheader("🎨 Dynamic Visualizations")
        
        # Dropdown allowing user to select ANY column dynamically
        plot_col = st.selectbox("Choose a column to visualize:", df.columns.tolist(), key="visual_select")
        clean_plot_data = df[plot_col].dropna()
        
        # Auto-detect data type to render correct chart
        if clean_plot_data.dtype == 'object' or clean_plot_data.nunique() < 10:
            # Categorical Data -> Render Bar Chart
            counts = clean_plot_data.value_counts().reset_index()
            counts.columns = [plot_col, 'Frequency']
            fig = px.bar(counts, x=plot_col, y='Frequency', 
                         title=f"Frequency Distribution of {plot_col}",
                         color='Frequency', color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Numerical Data -> Render Histogram and Box Plot side-by-side
            col_left, col_right = st.columns(2)
            with col_left:
                fig_hist = px.histogram(df, x=plot_col, title=f"Histogram of {plot_col}",
                                        marginal="rug", color_discrete_sequence=["#1f77b4"])
                st.plotly_chart(fig_hist, use_container_width=True)
            with col_right:
                fig_box = px.box(df, y=plot_col, title=f"Box Plot of {plot_col}",
                                 color_discrete_sequence=["#2ca02c"])
                st.plotly_chart(fig_box, use_container_width=True)

    # ----------------------------------------------------
    # TAB 2: AUTOMATED STATISTICAL TESTS
    # ----------------------------------------------------
    with tab2:
        st.subheader("🔬 Automated Hypothesis Testing Engine")
        
        test_col = st.selectbox("Choose a column to analyze mathematically:", df.columns.tolist(), key="test_select")
        col_data = df[test_col].dropna()
        
        # Test Routine A: Categorical Columns -> Chi-Squared Goodness of Fit Test
        if col_data.dtype == 'object' or col_data.nunique() < 10:
            st.markdown(f"### Chi-Squared Test on `{test_col}`")
            st.info("Hypothesis: H0 (Uniform Distribution) vs H1 (Not Uniform Distribution)")
            
            observed = col_data.value_counts()
            total_n = len(col_data)
            num_categories = len(observed)
            expected = np.full(num_categories, total_n / num_categories)
            
            chi2_stat, p_val = stats.chisquare(f_obs=observed, f_exp=expected)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Chi2 Statistic", f"{chi2_stat:.4f}")
            col2.metric("Degrees of Freedom (v)", num_categories - 1)
            col3.metric("P-Value", f"{p_val:.4f}")
            
            if p_val < 0.05:
                st.error("Result: Reject H0. There is statistically significant evidence that the data values are biased and not spread evenly.")
            else:
                st.success("Result: Fail to reject H0. The categories match a perfectly uniform distribution.")
                
        # Test Routine B: Continuous Numerical Columns -> 1-Sample T-Test
        else:
            st.markdown(f"### Hypothesis Test for the Mean on `{test_col}`")
            
            sample_mean = float(col_data.mean())
            sample_std = float(col_data.std(ddof=1))
            n = len(col_data)
            
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Sample Mean", f"{sample_mean:.4f}")
            col_s2.metric("Unbiased Std Dev", f"{sample_std:.4f}")
            col_s3.metric("Sample Size (n)", n)
            
            hypothesized_mu = st.number_input("Enter a hypothesized population mean:", value=float(round(sample_mean)))
            st.info(f"Hypothesis: H0 (Mean = {hypothesized_mu}) vs H1 (Mean is NOT equal to {hypothesized_mu})")
            
            t_stat, p_val = stats.ttest_1samp(col_data, hypothesized_mu)
            
            col_r1, col_r2 = st.columns(2)
            col_r1.metric("Calculated statistic (t)", f"{t_stat:.4f}")
            col_r2.metric("P-Value", f"{p_val:.4f}")
            
            if p_val < 0.05:
                st.error(f"Result: Reject H0. The true mean is significantly different from your hypothesized value of {hypothesized_mu} (at the 5% significance level).")
            else:
                st.success(f"Result: Fail to reject H0. There is no statistical evidence that the true mean differs from {hypothesized_mu}.")
