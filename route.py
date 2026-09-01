import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CLASSIFIER_DIR = BASE_DIR / "classifier"
TOPIC_DIR = BASE_DIR / "topic_model"
CLASSIFIER_MODEL_PATH = (CLASSIFIER_DIR / "classifier_model.onnx")
CLASSIFIER_VECTORIZER_PATH = (CLASSIFIER_DIR / "tfidf_vectorizer.pkl")
TOPIC_MODEL_PATH = (TOPIC_DIR / "data_modelling_model.onnx")
TOPIC_VECTORIZER_PATH = (TOPIC_DIR / "tfidf_vectorizer.pkl")
TOPIC_STATS_PATH = (TOPIC_DIR / "topic_stats.json")

# ============================================================
# MODEL CACHE
# ============================================================

_classifier_session = None
_classifier_vectorizer = None
_topic_session = None
_topic_vectorizer = None
_topic_statistics = None

# ============================================================
# LOAD CLASSIFIER
# ============================================================

def _load_classifier():

    global _classifier_session
    global _classifier_vectorizer

    # --------------------------------------------------------
    # Load ONNX model
    # --------------------------------------------------------

    if _classifier_session is None:
        if not CLASSIFIER_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Classifier ONNX model not found: "
                f"{CLASSIFIER_MODEL_PATH}"
            )

        _classifier_session = ort.InferenceSession(str(CLASSIFIER_MODEL_PATH),providers=["CPUExecutionProvider"])

    # --------------------------------------------------------
    # Load TF-IDF vectorizer
    # --------------------------------------------------------

    if _classifier_vectorizer is None:
        if not CLASSIFIER_VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                f"Classifier TF-IDF vectorizer not found: "
                f"{CLASSIFIER_VECTORIZER_PATH}"
            )

        _classifier_vectorizer = joblib.load(CLASSIFIER_VECTORIZER_PATH)

# ============================================================
# LOAD TOPIC MODEL
# ============================================================

def _load_topic_model():
    global _topic_session
    global _topic_vectorizer
    global _topic_statistics

    # --------------------------------------------------------
    # Load topic ONNX model
    # --------------------------------------------------------
    if _topic_session is None:
        if not TOPIC_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Topic ONNX model not found: "
                f"{TOPIC_MODEL_PATH}"
            )
        _topic_session = ort.InferenceSession(str(TOPIC_MODEL_PATH),providers=["CPUExecutionProvider"])

    # --------------------------------------------------------
    # Load topic TF-IDF
    # --------------------------------------------------------

    if _topic_vectorizer is None:
        if not TOPIC_VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                f"Topic TF-IDF vectorizer not found: "
                f"{TOPIC_VECTORIZER_PATH}"
            )

        _topic_vectorizer = joblib.load(TOPIC_VECTORIZER_PATH)

    # --------------------------------------------------------
    # Load topic statistics
    # --------------------------------------------------------

    if _topic_statistics is None:
        if not TOPIC_STATS_PATH.exists():
            raise FileNotFoundError(
                f"Topic statistics file not found: "
                f"{TOPIC_STATS_PATH}"
            )

        with open(TOPIC_STATS_PATH,"r",encoding="utf-8") as file:
            data = json.load(file)

        if "root" in data:
            _topic_statistics = data["root"]

        else:
            _topic_statistics = data

# ============================================================
# LOAD EVERYTHING
# ============================================================

def _load_all():

    _load_classifier()
    _load_topic_model()

# ============================================================
# CLASSIFIER
# ============================================================

def _predict_classifier(text):

    _load_classifier()

    # ========================================================
    # TEXT -> TF-IDF
    # ========================================================

    tfidf = _classifier_vectorizer.transform([text])

    # ONNX expects float32
    tfidf_dense = (tfidf.toarray().astype(np.float32))

    # ========================================================
    # GET INPUT NAME
    # ========================================================

    input_name = (_classifier_session.get_inputs()[0].name)

    # ========================================================
    # GET OUTPUT NAMES
    # ========================================================

    output_names = [output.name
        for output
        in _classifier_session.get_outputs()
    ]

    # ========================================================
    # RUN ONNX MODEL
    # ========================================================

    outputs = _classifier_session.run(output_names,{input_name: tfidf_dense})

    # ========================================================
    # MAP OUTPUT NAME -> OUTPUT VALUE
    # ========================================================

    output_map = {
        name: value
        for name, value
        in zip(output_names,outputs)
    }

    # ========================================================
    # EXTRACT DECISION SCORE
    # ========================================================

    if "decision_score" not in output_map:
        raise RuntimeError(
            "Classifier ONNX model does not contain "
            "'decision_score' output."
        )

    decision_score = float(np.asarray(output_map["decision_score"]).reshape(-1)[0])

    # ========================================================
    # EXTRACT PREDICTION
    # ========================================================

    if "prediction" not in output_map:
        raise RuntimeError(
            "Classifier ONNX model does not contain "
            "'prediction' output."
        )

    prediction = int(np.asarray(output_map["prediction"]).reshape(-1)[0])

    # ========================================================
    # CONVERT CLASS ID -> LABEL
    # ========================================================

    # Dataset convention:
    #
    #     0 = Fake
    #     1 = Real

    if prediction == 0:
        label = "fake"

    elif prediction == 1:
        label = "real"

    else:
        label = "unknown"

    return {
        "label":label,
        "class_id":prediction,
        "decision_score":decision_score
    }


# ============================================================
# TOPIC MODEL
# ============================================================

def _predict_topic(text):

    _load_topic_model()

    tfidf = _topic_vectorizer.transform([text])
    tfidf_dense = (tfidf.toarray().astype(np.float32))

    input_name = (_topic_session.get_inputs()[0].name)
    output_name = (_topic_session.get_outputs()[0].name)

    output = _topic_session.run([output_name],{input_name: tfidf_dense})[0]

    topic_distribution = (np.asarray(output[0]).astype(np.float64))

    total = topic_distribution.sum()

    if total <= 0:
        raise RuntimeError(
            "Topic model returned an invalid "
            "topic distribution."
        )

    topic_distribution = (topic_distribution / total)

    # ========================================================
    # DOMINANT TOPIC
    # ========================================================

    dominant_topic = int(np.argmax(topic_distribution))

    dominant_probability = float(topic_distribution[dominant_topic])

    topic_probabilities = []

    for topic_id, probability in enumerate(topic_distribution):
        topic_probabilities.append({
            "topic_id":topic_id,
            "probability":round(float(probability),6),
            "percentage":round(float(probability) * 100,2)
        })

    # ========================================================
    # LOOK UP TOPIC STATISTICS
    # ========================================================

    topic_key = str(dominant_topic)

    topic_info = (_topic_statistics.get(topic_key))

    # ========================================================
    # HANDLE MISSING TOPIC
    # ========================================================

    if topic_info is None:

        topic_info = {
            "topic_id":dominant_topic,
            "name":f"Topic {dominant_topic}",
            "top_words":[],
            "article_count":None,
            "fake_count":None,
            "real_count":None,
            "fake_percentage":None,
            "real_percentage":None,
            "reliability":"unknown"
        }

    # ========================================================
    # RETURN TOPIC RESULT
    # ========================================================

    return {
        "dominant_topic": {
            "topic_id":dominant_topic,
            "name":topic_info.get("name",f"Topic {dominant_topic}"),
            "probability":round(dominant_probability,6),
            "percentage":round(dominant_probability * 100,2),
            "top_words":topic_info.get("top_words",[])
        },
        "topic_distribution":topic_probabilities,
        "historical_statistics": {
            "article_count":topic_info.get("article_count"),
            "fake_count":topic_info.get("fake_count"),
            "real_count":topic_info.get("real_count"),
            "fake_percentage":topic_info.get("fake_percentage"),
            "real_percentage":topic_info.get("real_percentage"),
            "reliability":topic_info.get("reliability")
        }
    }


# ============================================================
# HEALTH CHECK
# ============================================================

def health():

    status = {
        "status":"ok",
        "classifier": {

            "model":
                False,

            "vectorizer":
                False
        },

        "topic_model": {

            "model":
                False,

            "vectorizer":
                False,

            "statistics":
                False
        }
    }

    errors = []

    # ========================================================
    # CLASSIFIER
    # ========================================================

    try:

        _load_classifier()

        status[
            "classifier"
        ][
            "model"
        ] = True

        status[
            "classifier"
        ][
            "vectorizer"
        ] = True

    except Exception as exc:

        errors.append({

            "component":
                "classifier",

            "error":
                str(exc)
        })

    # ========================================================
    # TOPIC MODEL
    # ========================================================

    try:

        _load_topic_model()

        status[
            "topic_model"
        ][
            "model"
        ] = True

        status[
            "topic_model"
        ][
            "vectorizer"
        ] = True

        status[
            "topic_model"
        ][
            "statistics"
        ] = True

    except Exception as exc:

        errors.append({

            "component":
                "topic_model",

            "error":
                str(exc)
        })

    # ========================================================
    # FINAL STATUS
    # ========================================================

    if errors:

        status["status"] = "degraded"

        status["errors"] = errors

    else:

        status["status"] = (
            "all systems operational"
        )

    return status


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict(text):
    """
    Main inference function.

    This is the function that app.py should call.

    It does NOT know how the models work internally.

    It simply:

        1. validates input
        2. runs classifier
        3. runs topic model
        4. combines results
        5. returns JSON-compatible data
    """

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    if text is None:

        raise ValueError(
            "Text cannot be None."
        )

    if not isinstance(
        text,
        str
    ):

        raise TypeError(
            "Text must be a string."
        )

    text = text.strip()

    if not text:

        raise ValueError(
            "Text cannot be empty."
        )

    # ========================================================
    # RUN CLASSIFIER
    # ========================================================

    classifier_result = (
        _predict_classifier(
            text
        )
    )

    # ========================================================
    # RUN TOPIC MODEL
    # ========================================================

    topic_result = (
        _predict_topic(
            text
        )
    )

    # ========================================================
    # BUILD FINAL RESPONSE
    # ========================================================

    response = {

        "success":
            True,

        # ----------------------------------------------------
        # Article information
        # ----------------------------------------------------

        "article": {

            "character_count":
                len(text),

            "word_count":
                len(
                    text.split()
                )
        },

        # ----------------------------------------------------
        # Fake / Real classification
        # ----------------------------------------------------

        "classification": {

            "prediction":
                classifier_result[
                    "label"
                ],

            "class_id":
                classifier_result[
                    "class_id"
                ],

            "decision_score":
                classifier_result[
                    "decision_score"
                ]
        },

        # ----------------------------------------------------
        # Dominant topic
        # ----------------------------------------------------

        "topic":
            topic_result[
                "dominant_topic"
            ],

        # ----------------------------------------------------
        # All 15 topic probabilities
        # ----------------------------------------------------

        "topic_distribution":
            topic_result[
                "topic_distribution"
            ],

        # ----------------------------------------------------
        # Historical statistics for dominant topic
        # ----------------------------------------------------

        "historical_topic_statistics":
            topic_result[
                "historical_statistics"
            ]
    }

    return response


# ============================================================
# DEBUG / INSPECTION
# ============================================================

def debug_models():
    """
    Print the input/output structure of both ONNX models.

    Useful during development to verify that the ONNX files
    contain the expected inputs and outputs.
    """

    _load_all()

    # ========================================================
    # CLASSIFIER
    # ========================================================

    print("\n" + "=" * 80)
    print("CLASSIFIER ONNX")
    print("=" * 80)

    print("\nInputs:")

    for item in _classifier_session.get_inputs():

        print(
            f"  Name   : {item.name}\n"
            f"  Shape  : {item.shape}\n"
            f"  Type   : {item.type}"
        )

    print("\nOutputs:")

    for item in _classifier_session.get_outputs():

        print(
            f"  Name   : {item.name}\n"
            f"  Shape  : {item.shape}\n"
            f"  Type   : {item.type}"
        )

    # ========================================================
    # TOPIC MODEL
    # ========================================================

    print("\n" + "=" * 80)
    print("TOPIC ONNX")
    print("=" * 80)

    print("\nInputs:")

    for item in _topic_session.get_inputs():

        print(
            f"  Name   : {item.name}\n"
            f"  Shape  : {item.shape}\n"
            f"  Type   : {item.type}"
        )

    print("\nOutputs:")

    for item in _topic_session.get_outputs():

        print(
            f"  Name   : {item.name}\n"
            f"  Shape  : {item.shape}\n"
            f"  Type   : {item.type}"
        )

