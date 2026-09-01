import requests
import pandas as pd
import streamlit as st

# ============================================================
# CONFIGURATION
# ============================================================

API_BASE_URL = "http://localhost:8000"

PREDICT_URL = f"{API_BASE_URL}/predict"
HEALTH_URL = f"{API_BASE_URL}/health"

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="News Truth & Topic Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .result-card {
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #ddd;
        background-color: white;
        margin-bottom: 15px;
    }

    .fake-card {
        border-left: 7px solid #d9534f;
    }

    .real-card {
        border-left: 7px solid #5cb85c;
    }

    .topic-card {
        border-left: 7px solid #4285f4;
    }

    .card-label {
        font-size: 13px;
        color: #777;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .card-value {
        font-size: 28px;
        font-weight: 700;
        margin-top: 5px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HELPER: API HEALTH
# ============================================================

def check_api_health():
    try:
        response = requests.get(HEALTH_URL,timeout=5)
        return (True,response.json())

    except requests.exceptions.ConnectionError:

        return ( False,
            {
                "status": "offline",
                "error":
                    "Cannot connect to FastAPI at "
                    "http://localhost:8000"
            }
        )

    except requests.exceptions.Timeout:
        return (False,
            {
                "status":"timeout",
                "error":"FastAPI request timed out."
            }
        )

    except Exception as exc:
        return (False,
            {
                "status":"error",
                "error":str(exc)
            }
        )

# ============================================================
# HELPER: PREDICT
# ============================================================

def call_prediction_api(text):
    try:
        response = requests.post(PREDICT_URL,json={"text": text},timeout=60)

        # ----------------------------------------------------
        # FastAPI error
        # ----------------------------------------------------

        if response.status_code != 200:
            try:
                error_data = (response.json())
                detail = error_data.get("detail","Unknown API error.")

            except Exception:
                detail = response.text

            return (False,{"error":detail})

        return (True,response.json())

    except requests.exceptions.ConnectionError:
        return (False,
            {
                "error":
                    "Cannot connect to FastAPI. "
                    "Make sure the backend is running "
                    "on localhost:8000."
            }
        )

    except requests.exceptions.Timeout:
        return (False,{"error":"Prediction request timed out."})

    except Exception as exc:
        return (False,{"error":str(exc)})

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("📰 News Analyzer")
    st.caption("Fake News Detection + Topic Modeling")

    st.divider()

    st.write("FastAPI provides the ML inference API.")

    st.write("Streamlit provides the user interface.")

# ============================================================
# PAGE NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "📰 Prediction",
        "❤️ System Health"
    ]
)

# ============================================================
# PAGE 1 — PREDICTION
# ============================================================

if page == "📰 Prediction":
    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        '<div class="main-title">'
        '📰 News Truth & Topic Analyzer'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
            Analyze a news article for Fake/Real
            classification and discover its topics.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    st.subheader("Enter News Article")

    article_text = st.text_area("Article text",height=300,placeholder=("Paste the complete news article here..."),label_visibility="collapsed")

    # --------------------------------------------------------
    # CHARACTER / WORD COUNT
    # --------------------------------------------------------

    if article_text:
        info1, info2 = st.columns(2)

        with info1:
            st.metric("Characters",len(article_text))

        with info2:
            st.metric("Words",len(article_text.split()))

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    analyze = st.button(
        "🔎 Analyze Article",
        type="primary",
        use_container_width=True
    )

    if analyze:
        if not article_text.strip():
            st.warning("Please enter an article first.")

        else:
            with st.spinner("Sending article to ML backend..."):
                success, result = (call_prediction_api(article_text))

            if not success:
                st.error(result.get("error","Prediction failed."))

            else:
                st.session_state["prediction_result"] = result

    # --------------------------------------------------------
    # DISPLAY SAVED RESULT
    # --------------------------------------------------------

    result = st.session_state.get("prediction_result")

    if result is None:
        st.info(
            "Enter an article and click "
            "'Analyze Article' to begin."
        )

    else:
        st.divider()
        st.header("Analysis Results")

        # ====================================================
        # EXTRACT RESULTS
        # ====================================================

        classification = result.get("classification",{})
        topic = result.get("topic",{})

        historical = result.get("historical_topic_statistics",{})

        topic_distribution = result.get("topic_distribution",[])

        prediction = classification.get("prediction","unknown")

        class_id = classification.get("class_id")

        decision_score = classification.get("decision_score")

        topic_name = topic.get("name","Unknown Topic")

        topic_percentage = topic.get("percentage",0)

        # ====================================================
        # TOP RESULT CARDS
        # ====================================================

        col1, col2, col3 = st.columns(3)

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        with col1:
            st.subheader("Classification")

            if prediction == "fake":
                st.error("🔴 FAKE")

            elif prediction == "real":
                st.success("🟢 REAL")

            else:
                st.warning("UNKNOWN")

        # ----------------------------------------------------
        # Primary Topic
        # ----------------------------------------------------

        with col2:

            st.subheader("Primary Topic")
            st.info(topic_name)

        # ----------------------------------------------------
        # Topic Relevance
        # ----------------------------------------------------

        with col3:

            st.subheader("Topic Relevance")
            st.metric("Relevance",f"{float(topic_percentage):.2f}%")

        # ====================================================
        # CLASSIFIER DETAILS
        # ====================================================

        st.subheader("Classification Details")
        c1, c2 = st.columns(2)

        with c1:
            st.metric("Class ID",(
                    class_id
                    if class_id is not None
                    else "N/A"
                )
            )

        with c2:
            if decision_score is not None:
                st.metric("SVM Decision Score",f"{float(decision_score):.4f}")

            else:
                st.metric("SVM Decision Score","N/A")

        st.caption(
            "The SVM decision score is not a probability. "
            "It represents the model's position relative "
            "to its classification decision boundary."
        )

        # ====================================================
        # TOPIC
        # ====================================================

        st.divider()
        st.header("Topic Analysis")
        topic_col1, topic_col2 = st.columns(2)

        # ----------------------------------------------------
        # Keywords
        # ----------------------------------------------------

        with topic_col1:
            st.subheader("Topic Keywords")
            top_words = topic.get("top_words",[])

            if top_words:
                for word in top_words:
                    st.markdown(f"`{word}`")

            else:
                st.write("No keywords available.")

        # ----------------------------------------------------
        # Topic relevance
        # ----------------------------------------------------

        with topic_col2:
            st.subheader("Primary Topic Relevance")
            progress_value = (float(topic_percentage)/ 100)
            progress_value = min(max(progress_value,0),1)

            st.progress(progress_value)

            st.write(f"**{float(topic_percentage):.2f}%**")

            st.caption(
                "This is the probability assigned "
                "by the topic model to the dominant topic."
            )

        # ====================================================
        # TOPIC DISTRIBUTION
        # ====================================================

        st.subheader("Topic Distribution")

        if topic_distribution:
            distribution_df = pd.DataFrame(topic_distribution)

            distribution_df = (distribution_df.sort_values("percentage",ascending=False))

            # ------------------------------------------------
            # Chart
            # ------------------------------------------------

            chart_df = (distribution_df[["topic_id","percentage"]].set_index("topic_id"))
            st.bar_chart(chart_df)

            # ------------------------------------------------
            # Table
            # ------------------------------------------------

            table_df = (distribution_df.copy())

            table_df.columns = [column.replace("_"," ").title()
                for column
                in table_df.columns
            ]

            st.dataframe(table_df,use_container_width=True,hide_index=True)

        # ====================================================
        # HISTORICAL TOPIC STATISTICS
        # ====================================================

        st.divider()

        st.header("Historical Statistics for This Topic")

        st.caption(
            "These statistics describe the Fake/Real "
            "distribution of historical articles "
            "associated with this topic."
        )

        article_count = historical.get("article_count")
        fake_count = historical.get("fake_count")
        real_count = historical.get("real_count")

        fake_percentage = historical.get("fake_percentage")
        real_percentage = historical.get("real_percentage")

        reliability = historical.get("reliability")

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        h1, h2, h3 = st.columns(3)
        with h1:
            st.metric("Total Articles",
                (
                    f"{article_count:,}"
                    if article_count is not None
                    else "N/A"
                )
            )

        with h2:
            st.metric("Fake Articles",
                (
                    f"{fake_count:,}"
                    if fake_count is not None
                    else "N/A"
                )
            )

        with h3:
            st.metric("Real Articles",
                (
                    f"{real_count:,}"
                    if real_count is not None
                    else "N/A"
                )
            )

        # ----------------------------------------------------
        # Historical chart
        # ----------------------------------------------------

        if (fake_percentage is not None and real_percentage is not None):

            historical_df = pd.DataFrame({"Percentage": [float(fake_percentage),float(real_percentage)]},index=["Fake","Real"])
            st.bar_chart(historical_df)

            p1, p2 = st.columns(2)

            with p1:
                st.metric("Historical Fake",f"{float(fake_percentage):.2f}%")

            with p2:
                st.metric("Historical Real",f"{float(real_percentage):.2f}%")

        # ----------------------------------------------------
        # Reliability
        # ----------------------------------------------------

        if reliability == "reliable":
            st.success("Reliable statistical sample")

        elif reliability == "limited":
            st.warning("Limited statistical sample")

        elif reliability == "very_low_sample":
            st.warning("Very low statistical sample")

        else:
            st.info(f"Reliability: {reliability}")

        # ====================================================
        # INTERPRETATION
        # ====================================================

        st.divider()

        st.header("Interpretation")

        st.markdown(
            f"""
            **Classification:** The trained TF-IDF + SVM
            classifier predicts this article as
            **{prediction.upper()}**.

            **Primary topic:** The topic model identifies
            **{topic_name}** as the dominant topic with
            **{float(topic_percentage):.2f}%** of the
            topic distribution.
            """
        )

        if (fake_percentage is not None and real_percentage is not None):

            st.markdown(
                f"""
                **Historical topic pattern:** In the
                historical dataset, articles associated
                with this topic were approximately
                **{float(fake_percentage):.2f}% Fake**
                and **{float(real_percentage):.2f}% Real**.
                """
            )

        st.warning(
            "Historical topic statistics are not a "
            "fact-checking verdict. They describe the "
            "distribution of labels in the dataset."
        )

# ============================================================
# PAGE 2 — HEALTH
# ============================================================

elif page == "System Health":
    st.markdown(
        '<div class="main-title">'
        '❤️ System Health'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
            Check the availability of the FastAPI
            backend and all ML artifacts.
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("🔄 Check System",type="primary",use_container_width=True):
        with st.spinner("Checking backend..."):
            api_available, health_result = (check_api_health())

        st.session_state["health_result"] = health_result
        st.session_state["api_available"] = api_available

    health_result = st.session_state.get("health_result")
    api_available = st.session_state.get("api_available",False)

    if health_result is None:
        st.info(
            "Click 'Check System' to check the "
            "FastAPI backend."
        )

    else:
        st.header("FastAPI Status")

        if api_available:
            st.success("🟢 FastAPI is reachable")

        else:
            st.error("🔴 FastAPI is not reachable")
            st.warning(health_result.get("error","Unknown error."))

        if api_available:
            status = health_result.get("status","unknown")

            if status == "all systems operational":
                st.success("✓ All systems operational")

            elif status == "degraded":
                st.warning("⚠ System is degraded")

            else:
                st.error(f"System status: {status}")

            st.divider()
            st.header("Classifier")

            classifier = (
                health_result.get(
                    "classifier",
                    {}
                )
            )

            c1, c2 = st.columns(2)

            with c1:

                if classifier.get(
                    "model"
                ):

                    st.success(
                        "✓ classifier_model.onnx"
                    )

                else:

                    st.error(
                        "✗ classifier_model.onnx"
                    )

            with c2:

                if classifier.get(
                    "vectorizer"
                ):

                    st.success(
                        "✓ TF-IDF vectorizer"
                    )

                else:

                    st.error(
                        "✗ TF-IDF vectorizer"
                    )

            # =================================================
            # TOPIC MODEL
            # =================================================

            st.divider()

            st.header(
                "Topic Model"
            )

            topic_model = (
                health_result.get(
                    "topic_model",
                    {}
                )
            )

            t1, t2, t3 = st.columns(3)

            with t1:

                if topic_model.get(
                    "model"
                ):

                    st.success(
                        "✓ data_modelling_model.onnx"
                    )

                else:

                    st.error(
                        "✗ Topic ONNX model"
                    )

            with t2:

                if topic_model.get(
                    "vectorizer"
                ):

                    st.success(
                        "✓ TF-IDF vectorizer"
                    )

                else:

                    st.error(
                        "✗ TF-IDF vectorizer"
                    )

            with t3:

                if topic_model.get(
                    "statistics"
                ):

                    st.success(
                        "✓ topic_stats.json"
                    )

                else:

                    st.error(
                        "✗ topic_stats.json"
                    )

            # =================================================
            # ERRORS
            # =================================================

            errors = health_result.get(
                "errors",
                []
            )

            if errors:

                st.divider()

                st.header(
                    "Errors"
                )

                for error in errors:

                    component = error.get(
                        "component",
                        "unknown"
                    )

                    message = error.get(
                        "error",
                        "Unknown error"
                    )

                    st.error(
                        f"{component}: {message}"
                    )



# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "News Truth & Topic Analyzer • "
    "FastAPI + Streamlit • "
    "TF-IDF + SVM + LDA • ONNX"
)