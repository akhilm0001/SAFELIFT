import os
import csv
import json
import math
import threading
from datetime import datetime

import cv2
import mediapipe as mp
import pandas as pd
import streamlit as st

# ============================================================
# SAFELIFT AI
# ============================================================

st.set_page_config(
    page_title="SafeLift AI",
    page_icon="🦺",
    layout="wide"
)

st.title("🦺 SafeLift AI")
st.subheader("Computer Vision-Based Lifting Posture Assessment")

CSV_FILE = "safelift_reba_records.csv"
CONFIG_FILE = "safelift_config.json"
SCREENSHOT_FOLDER = "safelift_screenshots"

os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)

HEADERS = [
    "Time",
    "Cycle_Number",
    "Max_Torso_Angle_Deg",
    "Min_Knee_Angle_Deg",
    "Neck_Angle_Deg",
    "Upper_Arm_Angle_Deg",
    "Elbow_Angle_Deg",
    "Cycle_Duration_Seconds",
    "REBA_Score",
    "REBA_Action_Level"
]

STAGE_NAMES = [
    "01_upright",
    "02_start_bend",
    "03_middle_bend",
    "04_lowest",
    "05_return"
]

DEFAULT_CONFIG = {
    "task_name": "Repetitive ground-level lifting",
    "load_weight_kg": 15.0,
    "shock_or_rapid_force": False,
    "coupling_quality": "good",
    "trunk_twist": False,
    "trunk_side_bending": False,
    "neck_twist": False,
    "neck_side_bending": False,
    "wrist_angle_deg": 0,
    "wrist_twist": False,
    "arm_abducted": False,
    "shoulder_raised": False,
    "arm_supported": False,
    "one_leg_weight_bearing": False,
    "static_posture_over_1_minute": False,
    "repetitive_small_range_actions": False,
    "rapid_posture_changes_or_unstable_base": False
}

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


def config_to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in (
            "true", "yes", "y", "1"
        )
    return False


def load_config():
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError):
        config = DEFAULT_CONFIG.copy()

    for key, value in DEFAULT_CONFIG.items():
        config.setdefault(key, value)

    return config


# ============================================================
# REBA TABLES AND SCORING
# ============================================================

TABLE_A = {
    1: {
        1: [1, 2, 3, 4],
        2: [1, 2, 3, 4],
        3: [3, 3, 5, 6]
    },
    2: {
        1: [2, 3, 4, 5],
        2: [3, 4, 5, 6],
        3: [4, 5, 6, 7]
    },
    3: {
        1: [2, 4, 5, 6],
        2: [4, 5, 6, 7],
        3: [5, 6, 7, 8]
    },
    4: {
        1: [3, 5, 6, 7],
        2: [5, 6, 7, 8],
        3: [6, 7, 8, 9]
    },
    5: {
        1: [4, 6, 7, 8],
        2: [6, 7, 8, 9],
        3: [7, 8, 9, 9]
    }
}

TABLE_B = {
    1: {
        1: [1, 2, 2],
        2: [1, 2, 3]
    },
    2: {
        1: [1, 2, 3],
        2: [2, 3, 4]
    },
    3: {
        1: [3, 4, 5],
        2: [4, 5, 5]
    },
    4: {
        1: [4, 5, 5],
        2: [5, 6, 7]
    },
    5: {
        1: [6, 7, 8],
        2: [7, 8, 8]
    },
    6: {
        1: [7, 8, 8],
        2: [8, 9, 9]
    }
}

TABLE_C = [
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
    [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
]


def calculate_reba(
    torso_angle,
    knee_angle,
    neck_angle,
    upper_arm_angle,
    elbow_angle,
    load_kg,
    shock_force,
    coupling_quality,
    trunk_twist=False,
    trunk_side_bending=False,
    neck_twist=False,
    neck_side_bending=False,
    wrist_angle=0,
    wrist_twist=False,
    arm_abducted=False,
    shoulder_raised=False,
    arm_supported=False,
    one_leg_weight_bearing=False,
    static_posture=False,
    repetitive_action=False,
    rapid_or_unstable=False
):
    torso_angle = abs(float(torso_angle))
    knee_angle = float(knee_angle)
    neck_angle = abs(float(neck_angle))
    upper_arm_angle = abs(float(upper_arm_angle))
    elbow_angle = abs(float(elbow_angle))
    wrist_angle = abs(float(wrist_angle))
    load_kg = float(load_kg)

    if torso_angle <= 5:
        trunk_score = 1
    elif torso_angle <= 20:
        trunk_score = 2
    elif torso_angle <= 60:
        trunk_score = 3
    else:
        trunk_score = 4

    trunk_score = min(
        trunk_score
        + int(config_to_bool(trunk_twist))
        + int(config_to_bool(trunk_side_bending)),
        5
    )

    neck_score = 1 if neck_angle <= 20 else 2
    neck_score = min(
        neck_score
        + int(config_to_bool(neck_twist))
        + int(config_to_bool(neck_side_bending)),
        3
    )

    knee_flexion = max(0, 180 - knee_angle)
    leg_score = 1

    if knee_flexion > 60:
        leg_score += 2
    elif knee_flexion >= 30:
        leg_score += 1

    if config_to_bool(one_leg_weight_bearing):
        leg_score += 1

    leg_score = min(leg_score, 4)

    posture_score_a = TABLE_A[trunk_score][neck_score][leg_score - 1]

    if load_kg < 5:
        load_score = 0
    elif load_kg <= 10:
        load_score = 1
    else:
        load_score = 2

    if config_to_bool(shock_force):
        load_score += 1

    score_a = min(posture_score_a + load_score, 12)

    if upper_arm_angle <= 20:
        upper_arm_score = 1
    elif upper_arm_angle <= 45:
        upper_arm_score = 2
    elif upper_arm_angle <= 90:
        upper_arm_score = 3
    else:
        upper_arm_score = 4

    upper_arm_score += int(config_to_bool(arm_abducted))
    upper_arm_score += int(config_to_bool(shoulder_raised))
    upper_arm_score -= int(config_to_bool(arm_supported))
    upper_arm_score = max(1, min(upper_arm_score, 6))

    lower_arm_score = (
        1 if 60 <= elbow_angle <= 100 else 2
    )

    wrist_score = 1 if wrist_angle <= 15 else 2

    if config_to_bool(wrist_twist):
        wrist_score += 1

    wrist_score = min(wrist_score, 3)

    posture_score_b = TABLE_B[upper_arm_score][lower_arm_score][wrist_score - 1]

    coupling_scores = {
        "good": 0,
        "fair": 1,
        "poor": 2,
        "unacceptable": 3
    }

    coupling_key = str(coupling_quality).strip().lower()

    if coupling_key not in coupling_scores:
        raise ValueError(
            "Coupling must be good, fair, poor, or unacceptable."
        )

    score_b = min(
        posture_score_b + coupling_scores[coupling_key],
        12
    )

    score_c = TABLE_C[score_a - 1][score_b - 1]

    activity_score = (
        int(config_to_bool(static_posture))
        + int(config_to_bool(repetitive_action))
        + int(config_to_bool(rapid_or_unstable))
    )

    final_score = min(score_c + activity_score, 15)

    if final_score == 1:
        level = "Negligible"
    elif final_score <= 3:
        level = "Low"
    elif final_score <= 7:
        level = "Medium"
    elif final_score <= 10:
        level = "High"
    else:
        level = "Very High"

    return final_score, level


def add_risk_explanations(df):
    df = df.copy()

    aliases = {
        "torso": [
            "Max_Torso_Angle_Deg", "Torso_Angle_Deg", "Torso °"
        ],
        "knee": [
            "Min_Knee_Angle_Deg", "Knee_Angle_Deg", "Knee °"
        ],
        "neck": [
            "Neck_Angle_Deg",
            "Max_Neck_Flexion_Estimate_Deg",
            "Neck °"
        ],
        "arm": [
            "Upper_Arm_Angle_Deg",
            "Max_Upper_Arm_Angle_Deg",
            "Arm °"
        ],
        "elbow": [
            "Elbow_Angle_Deg",
            "Min_Elbow_Angle_Deg",
            "Elbow °"
        ],
        "score": ["REBA_Score", "REBA"],
        "level": [
            "REBA_Action_Level", "Risk level", "Action_Level"
        ]
    }

    def get_value(row, key):
        for col in aliases[key]:
            if col in row.index:
                value = pd.to_numeric(row[col], errors="coerce")
                if pd.notna(value):
                    return float(value)
        return None

    def explain(row):
        findings = []
        precautions = []

        torso = get_value(row, "torso")
        knee = get_value(row, "knee")
        neck = get_value(row, "neck")
        arm = get_value(row, "arm")
        elbow = get_value(row, "elbow")
        score = get_value(row, "score")

        level = None
        for col in aliases["level"]:
            if col in row.index and pd.notna(row[col]):
                text_value = str(row[col]).strip()
                if text_value:
                    level = text_value
                    break

        if torso is not None:
            if torso > 60:
                findings.append("marked forward torso bending")
                precautions.append(
                    "Reduce trunk bending. Move closer to the load "
                    "and raise the work surface where practical."
                )
            elif torso > 20:
                findings.append("forward torso inclination")
                precautions.append(
                    "Keep the load close and avoid reaching forward."
                )

        if knee is not None:
            flexion = 180 - knee
            if flexion > 60:
                findings.append("deep knee flexion or squatting")
                precautions.append(
                    "Use a stable stance and adjust the work height "
                    "to avoid excessive squatting where possible."
                )
            elif flexion >= 30:
                findings.append("moderate knee flexion")
                precautions.append(
                    "Keep your footing stable and avoid twisting "
                    "while rising."
                )

        if neck is not None and neck > 20:
            findings.append("increased neck-angle estimate")
            precautions.append(
                "Keep the head and neck comfortable and neutral; "
                "improve visibility of the load."
            )

        if arm is not None and arm > 45:
            findings.append("elevated upper-arm posture")
            precautions.append(
                "Keep elbows closer to the body and reduce reaching."
            )

        if elbow is not None and not (60 <= elbow <= 100):
            findings.append(
                "elbow angle outside the preferred working range"
            )
            precautions.append(
                "Adjust the load distance and hand position where possible."
            )

        if not findings:
            findings.append(
                "no angle threshold flagged from the available measurements"
            )

        if level:
            why = (
                f"Recorded risk level: {level}. "
                + "; ".join(findings)
                + "."
            )
        elif score is not None:
            why = (
                f"Recorded REBA score: {score:g}. "
                + "; ".join(findings)
                + "."
            )
        else:
            why = (
                "; ".join(findings).capitalize()
                + ". Risk level cannot be determined from available fields."
            )

        if not precautions:
            precautions.append(
                "Keep the load close, avoid twisting, maintain stable footing, "
                "and review the full task setup and load characteristics."
            )

        return why, " ".join(dict.fromkeys(precautions))

    explanations = df.apply(explain, axis=1)

    df["Why_This_Risk"] = [x[0] for x in explanations]
    df["Precautions"] = [x[1] for x in explanations]

    return df


def render_results(df, prefix="safelift_results"):
    if df is None or df.empty:
        st.info("No completed lifts to display yet.")
        return

    enriched = add_risk_explanations(df)

    display_map = {
        "Time": "Time",
        "Cycle_Number": "Lift",
        "Max_Torso_Angle_Deg": "Torso °",
        "Min_Knee_Angle_Deg": "Knee °",
        "Neck_Angle_Deg": "Neck °",
        "Upper_Arm_Angle_Deg": "Arm °",
        "Elbow_Angle_Deg": "Elbow °",
        "Cycle_Duration_Seconds": "Duration (s)",
        "REBA_Score": "REBA",
        "REBA_Action_Level": "Risk level",
        "Why_This_Risk": "Why this risk?",
        "Precautions": "Precautions"
    }

    preferred = [
        "Time", "Cycle_Number",
        "Max_Torso_Angle_Deg", "Min_Knee_Angle_Deg",
        "Neck_Angle_Deg", "Upper_Arm_Angle_Deg",
        "Elbow_Angle_Deg", "Cycle_Duration_Seconds",
        "REBA_Score", "REBA_Action_Level",
        "Why_This_Risk", "Precautions"
    ]

    ordered = [c for c in preferred if c in enriched.columns]
    ordered += [c for c in enriched.columns if c not in ordered]

    shown = enriched[ordered].rename(columns=display_map)

    scores = pd.to_numeric(
        enriched.get("REBA_Score", pd.Series(dtype=float)),
        errors="coerce"
    )

    levels = enriched.get(
        "REBA_Action_Level",
        pd.Series("Unknown", index=enriched.index)
    ).astype(str)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🧍 Completed lifts", len(enriched))
    c2.metric(
        "⚠️ Highest REBA",
        int(scores.max()) if scores.notna().any() else "—"
    )
    c3.metric(
        "📊 Average REBA",
        f"{scores.mean():.1f}" if scores.notna().any() else "—"
    )
    c4.metric(
        "🔴 High / very high",
        int(levels.str.lower().isin(["high", "very high"]).sum())
    )

    st.subheader("📋 Lift-by-lift results")

    # EDIT: Use st.table instead of st.dataframe so long text is easier to read.
    st.table(shown)

    st.download_button(
        "📥 Download explained results (CSV)",
        enriched.to_csv(index=False).encode("utf-8-sig"),
        f"{prefix}.csv",
        "text/csv",
        key=prefix + "_download"
    )

    if "REBA_Score" in enriched.columns:
        chart = enriched[
            [c for c in ["Cycle_Number", "REBA_Score"]
             if c in enriched.columns]
        ].copy()

        if len(chart.columns) == 2:
            chart["REBA_Score"] = pd.to_numeric(
                chart["REBA_Score"], errors="coerce"
            )
            st.subheader("📈 REBA score by lift")
            st.line_chart(chart.set_index("Cycle_Number"))

    st.subheader("💡 Detailed explanation")

    idx = st.selectbox(
        "Select a lift",
        list(range(len(enriched))),
        format_func=lambda i: (
            f"Lift {enriched.iloc[i].get('Cycle_Number', i + 1)} — "
            f"{enriched.iloc[i].get('REBA_Action_Level', 'Risk not recorded')}"
        ),
        key=prefix + "_detail"
    )

    row = enriched.iloc[idx]

    st.markdown("**Why this risk?**")
    st.write(row["Why_This_Risk"])

    st.markdown("**🛡️ Precautions**")
    st.write(row["Precautions"])


def ensure_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADERS)
    else:
        try:
            existing = pd.read_csv(CSV_FILE, nrows=0).columns.tolist()
        except Exception:
            existing = []

        if existing != HEADERS:
            st.error(
                f"{CSV_FILE} has different headers. Back up or rename it "
                "before starting this app."
            )
            st.stop()


def append_record(record):
    ensure_csv()
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writerow(record)


def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


def get_point(landmarks, landmark, width, height):
    point = landmarks[landmark]
    return int(point.x * width), int(point.y * height)


def process_pose_frame(frame, pose):
    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    annotated = frame.copy()
    measurements = None

    if not result.pose_landmarks:
        cv2.putText(
            annotated, "Move into full camera view", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        return annotated, None

    landmarks = result.pose_landmarks.landmark
    mp_draw.draw_landmarks(
        annotated, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
    )

    required = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST
    ]

    if not all(landmarks[p.value].visibility > 0.5 for p in required):
        cv2.putText(
            annotated,
            "Turn sideways; keep your full body visible",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )
        return annotated, None

    shoulder = get_point(
        landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, width, height
    )
    hip = get_point(
        landmarks, mp_pose.PoseLandmark.LEFT_HIP, width, height
    )
    knee = get_point(
        landmarks, mp_pose.PoseLandmark.LEFT_KNEE, width, height
    )
    ankle = get_point(
        landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, width, height
    )
    ear = get_point(
        landmarks, mp_pose.PoseLandmark.LEFT_EAR, width, height
    )
    elbow = get_point(
        landmarks, mp_pose.PoseLandmark.LEFT_ELBOW, width, height
    )
    wrist = get_point(
        landmarks, mp_pose.PoseLandmark.LEFT_WRIST, width, height
    )

    knee_angle = calculate_angle(hip, knee, ankle)

    dx = shoulder[0] - hip[0]
    dy = hip[1] - shoulder[1]

    torso_angle = math.degrees(
        math.atan2(abs(dx), max(abs(dy), 1))
    )

    neck_angle = abs(180.0 - calculate_angle(ear, shoulder, hip))
    upper_arm_angle = calculate_angle(hip, shoulder, elbow)
    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    measurements = {
        "torso": torso_angle,
        "knee": knee_angle,
        "neck": neck_angle,
        "upper_arm": upper_arm_angle,
        "elbow": elbow_angle
    }

    return annotated, measurements


class LiftCycleTracker:
    BEND_ANGLE = 45
    RETURN_ANGLE = 20
    KNEE_BEND_ANGLE = 145
    UPRIGHT_KNEE_ANGLE = 170
    UPRIGHT_CONFIRM_FRAMES = 5
    MOVEMENT_CONFIRM_FRAMES = 5
    MIN_CYCLE_SECONDS = 0.8

    def __init__(self, config, persist=True, start_count=0):
        self.config = config
        self.persist = persist
        self.lock = threading.Lock()
        self.torso_history = []
        self.knee_history = []
        self.cycle_active = False
        self.ready_for_lift = False
        self.movement_frames = 0
        self.upright_return_frames = 0
        self.cycle_start = None
        # Every new session starts from 0, not from the old CSV rows.
        self.lift_count = int(start_count)
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.upright_frame = None
        self.early_frame = None
        self.middle_frame = None
        self.lowest_frame = None
        self.max_torso = 0.0
        self.min_knee = 180.0
        self.max_neck = 0.0
        self.max_arm = 0.0
        self.max_elbow = 0.0
        self.lowest_angle = 0.0
        self.latest_stage = "Waiting for upright posture"
        self.latest_angles = {}
        self.latest_score = None
        self.latest_level = ""
        self.last_record = None

    def _existing_count(self):
        try:
            return int(pd.read_csv(CSV_FILE)["Cycle_Number"].count())
        except Exception:
            return 0

    def _reset_cycle(self, frame):
        self.cycle_active = False
        self.cycle_start = None
        self.ready_for_lift = True
        self.movement_frames = 0
        self.upright_return_frames = 0
        self.upright_frame = frame.copy()
        self.early_frame = None
        self.middle_frame = None
        self.lowest_frame = None
        self.max_torso = 0.0
        self.min_knee = 180.0
        self.max_neck = 0.0
        self.max_arm = 0.0
        self.max_elbow = 0.0
        self.lowest_angle = 0.0

    def update(self, frame, m):
        now = datetime.now()

        self.torso_history.append(m["torso"])
        self.knee_history.append(m["knee"])

        self.torso_history = self.torso_history[-5:]
        self.knee_history = self.knee_history[-5:]

        torso = sorted(self.torso_history)[len(self.torso_history) // 2]
        knee = sorted(self.knee_history)[len(self.knee_history) // 2]

        neck = m["neck"]
        arm = m["upper_arm"]
        elbow = m["elbow"]

        movement_started = (
            torso >= self.BEND_ANGLE or knee <= self.KNEE_BEND_ANGLE
        )

        upright_now = (
            torso < self.RETURN_ANGLE
            and knee >= self.UPRIGHT_KNEE_ANGLE
        )

        stage = (
            "UPRIGHT" if upright_now
            else "BENDING / SQUATTING" if movement_started
            else "MOVING"
        )

        if not self.cycle_active and upright_now:
            self.ready_for_lift = True
            if self.upright_frame is None:
                self.upright_frame = frame.copy()

        if not self.cycle_active and self.ready_for_lift and movement_started:
            self.movement_frames += 1
        elif not self.cycle_active:
            self.movement_frames = 0

        if (
            self.ready_for_lift
            and not self.cycle_active
            and self.movement_frames >= self.MOVEMENT_CONFIRM_FRAMES
        ):
            self.cycle_active = True
            self.cycle_start = now
            self.max_torso = torso
            self.min_knee = knee
            self.max_neck = neck
            self.max_arm = arm
            self.max_elbow = elbow
            self.lowest_angle = torso
            self.early_frame = frame.copy()
            self.middle_frame = None
            self.lowest_frame = frame.copy()
            self.upright_return_frames = 0
            self.movement_frames = 0
            self.ready_for_lift = False
            stage = "LIFT STARTED"

        completed = None

        if self.cycle_active:
            self.max_torso = max(self.max_torso, torso)
            self.min_knee = min(self.min_knee, knee)
            self.max_neck = max(self.max_neck, neck)
            self.max_arm = max(self.max_arm, arm)
            self.max_elbow = max(self.max_elbow, elbow)

            if 55 <= torso < 70 and self.middle_frame is None:
                self.middle_frame = frame.copy()

            if torso > self.lowest_angle:
                self.lowest_angle = torso
                self.lowest_frame = frame.copy()

            if upright_now:
                self.upright_return_frames += 1
            else:
                self.upright_return_frames = 0

            stage = (
                "RETURNING UPRIGHT" if upright_now else "LIFT IN PROGRESS"
            )

            if self.upright_return_frames >= self.UPRIGHT_CONFIRM_FRAMES:
                duration = (now - self.cycle_start).total_seconds()

                if duration >= self.MIN_CYCLE_SECONDS:
                    score, level = calculate_reba(
                        torso_angle=self.max_torso,
                        knee_angle=self.min_knee,
                        neck_angle=self.max_neck,
                        upper_arm_angle=self.max_arm,
                        elbow_angle=self.max_elbow,
                        load_kg=float(self.config["load_weight_kg"]),
                        shock_force=self.config["shock_or_rapid_force"],
                        coupling_quality=self.config["coupling_quality"],
                        trunk_twist=self.config["trunk_twist"],
                        trunk_side_bending=self.config["trunk_side_bending"],
                        neck_twist=self.config["neck_twist"],
                        neck_side_bending=self.config["neck_side_bending"],
                        wrist_angle=self.config["wrist_angle_deg"],
                        wrist_twist=self.config["wrist_twist"],
                        arm_abducted=self.config["arm_abducted"],
                        shoulder_raised=self.config["shoulder_raised"],
                        arm_supported=self.config["arm_supported"],
                        one_leg_weight_bearing=self.config["one_leg_weight_bearing"],
                        static_posture=self.config["static_posture_over_1_minute"],
                        repetitive_action=self.config["repetitive_small_range_actions"],
                        rapid_or_unstable=self.config["rapid_posture_changes_or_unstable_base"]
                    )

                    self.lift_count += 1

                    completed = {
                        "Time": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "Cycle_Number": self.lift_count,
                        "Max_Torso_Angle_Deg": round(self.max_torso, 1),
                        "Min_Knee_Angle_Deg": round(self.min_knee, 1),
                        "Neck_Angle_Deg": round(self.max_neck, 1),
                        "Upper_Arm_Angle_Deg": round(self.max_arm, 1),
                        "Elbow_Angle_Deg": round(self.max_elbow, 1),
                        "Cycle_Duration_Seconds": round(duration, 2),
                        "REBA_Score": score,
                        "REBA_Action_Level": level
                    }

                    if self.persist:
                        append_record(completed)

                    self.last_record = completed
                    self.latest_score = score
                    self.latest_level = level

                    cycle_dir = os.path.join(
                        SCREENSHOT_FOLDER,
                        f"session_{self.session_id}",
                        f"cycle_{self.lift_count:03d}"
                    )
                    os.makedirs(cycle_dir, exist_ok=True)

                    images = [
                        self.upright_frame,
                        self.early_frame,
                        self.middle_frame,
                        self.lowest_frame,
                        frame
                    ]

                    for i, image in enumerate(images):
                        if image is not None:
                            cv2.imwrite(
                                os.path.join(
                                    cycle_dir, STAGE_NAMES[i] + ".jpg"
                                ),
                                image
                            )

                self._reset_cycle(frame)
                stage = "LIFT COMPLETED"

        self.latest_stage = stage
        self.latest_angles = {
            "Torso angle (°)": torso,
            "Knee angle (°)": knee,
            "Neck angle (°)": neck,
            "Upper-arm angle (°)": arm,
            "Elbow angle (°)": elbow
        }

        return completed, stage


# ============================================================
# APPLICATION SETTINGS
# ============================================================

ensure_csv()
config = load_config()

with st.sidebar:
    st.header("⚙️ Task settings")
    st.caption("Settings apply to new lift records.")

    config["task_name"] = st.text_input(
        "Task name", str(config["task_name"])
    )

    config["load_weight_kg"] = st.number_input(
        "Load weight (kg)",
        min_value=0.0,
        value=float(config["load_weight_kg"]),
        step=0.5
    )

    choices = ["good", "fair", "poor", "unacceptable"]
    current = str(config["coupling_quality"]).lower()
    config["coupling_quality"] = st.selectbox(
        "Coupling quality",
        choices,
        index=choices.index(current) if current in choices else 0
    )

    config["shock_or_rapid_force"] = st.checkbox(
        "Shock or rapid force", bool(config["shock_or_rapid_force"])
    )
    config["trunk_twist"] = st.checkbox(
        "Trunk twist", bool(config["trunk_twist"])
    )
    config["trunk_side_bending"] = st.checkbox(
        "Trunk side bending", bool(config["trunk_side_bending"])
    )
    config["neck_twist"] = st.checkbox(
        "Neck twist", bool(config["neck_twist"])
    )
    config["neck_side_bending"] = st.checkbox(
        "Neck side bending", bool(config["neck_side_bending"])
    )

    config["wrist_angle_deg"] = st.number_input(
        "Wrist angle (degrees)",
        min_value=0.0,
        max_value=180.0,
        value=float(config["wrist_angle_deg"]),
        step=1.0
    )

    config["wrist_twist"] = st.checkbox(
        "Wrist twist", bool(config["wrist_twist"])
    )
    config["arm_abducted"] = st.checkbox(
        "Arm abducted", bool(config["arm_abducted"])
    )
    config["shoulder_raised"] = st.checkbox(
        "Shoulder raised", bool(config["shoulder_raised"])
    )
    config["arm_supported"] = st.checkbox(
        "Arm supported", bool(config["arm_supported"])
    )
    config["one_leg_weight_bearing"] = st.checkbox(
        "One-leg weight bearing",
        bool(config["one_leg_weight_bearing"])
    )
    config["static_posture_over_1_minute"] = st.checkbox(
        "Static posture over 1 minute",
        bool(config["static_posture_over_1_minute"])
    )
    config["repetitive_small_range_actions"] = st.checkbox(
        "Repetitive small-range actions",
        bool(config["repetitive_small_range_actions"])
    )
    config["rapid_posture_changes_or_unstable_base"] = st.checkbox(
        "Rapid posture changes / unstable base",
        bool(config["rapid_posture_changes_or_unstable_base"])
    )

    if st.button("💾 Save task settings", use_container_width=True):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        st.success("Task settings saved.")


mode = st.sidebar.radio(
    "Choose mode",
    ["Live camera", "Upload video", "Upload CSV", "View lift records"]
)


# ============================================================
# LIVE CAMERA MODE
# ============================================================

if mode == "Live camera":
    st.header("📷 Live lift-cycle detection")
    st.write(
        "Allow camera access. Stand sideways with your full body visible. "
        "Start upright, perform the lift, and return upright."
    )

    try:
        from streamlit_webrtc import (
            webrtc_streamer, VideoProcessorBase, WebRtcMode
        )
        from av import VideoFrame

        class SafeLiftVideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.pose = mp_pose.Pose(
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                # New tracker per camera start -> lift count begins at 0
                self.tracker = LiftCycleTracker(
                    config.copy(), start_count=0
                )
                self.lock = threading.Lock()
                self.last_error = None

            def recv(self, frame):
                image = frame.to_ndarray(format="bgr24")
                image = cv2.flip(image, 1)

                try:
                    annotated, measurements = process_pose_frame(
                        image, self.pose
                    )

                    with self.lock:
                        if measurements:
                            self.tracker.update(annotated, measurements)
                            values = self.tracker.latest_angles
                            count = self.tracker.lift_count
                            score = self.tracker.latest_score
                            level = self.tracker.latest_level

                            for i, (key, value) in enumerate(values.items()):
                                cv2.putText(
                                    annotated,
                                    f"{key}: {value:.1f}",
                                    (15, 30 + i * 25),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (0, 255, 0), 2
                                )

                            cv2.putText(
                                annotated,
                                f"Stage: {self.tracker.latest_stage}",
                                (15, 180),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 165, 255), 2
                            )
                            cv2.putText(
                                annotated,
                                f"Completed lifts: {count}",
                                (15, 210),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 0), 2
                            )

                            if score is not None:
                                cv2.putText(
                                    annotated,
                                    f"Last REBA: {score} ({level})",
                                    (15, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 0, 255), 2
                                )
                        else:
                            cv2.putText(
                                annotated,
                                "Waiting for visible body",
                                (15, 70),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2
                            )

                except Exception as e:
                    self.last_error = str(e)
                    cv2.putText(
                        annotated, "Processing error - check app",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2
                    )

                return VideoFrame.from_ndarray(
                    annotated, format="bgr24"
                )

        view_col1, view_col2 = st.columns([3, 1])
        with view_col1:
            view_width = st.slider(
                "Camera view size (zoom)",
                min_value=40, max_value=100, value=100, step=5,
                format="%d%%",
                key="camera_view_width"
            )
        with view_col2:
            fit_screen = st.checkbox(
                "Fit to screen height",
                value=False,
                key="camera_fit_screen",
                help="Limits the video height so the whole picture "
                     "stays visible without scrolling."
            )

        video_style = {
            "width": f"{view_width}%",
            "display": "block",
            "margin": "0 auto",
            "objectFit": "contain"
        }
        if fit_screen:
            video_style["maxHeight"] = "70vh"
            video_style["width"] = "auto"
            video_style["maxWidth"] = f"{view_width}%"

        webrtc_ctx = webrtc_streamer(
            key="safelift-live-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=SafeLiftVideoProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280, "min": 640},
                    "height": {"ideal": 720, "min": 480},
                    "frameRate": {"ideal": 30}
                },
                "audio": False
            },
            video_html_attrs={
                "style": video_style,
                # controls=True adds the browser's full-screen button
                "controls": True,
                "autoPlay": True,
                "muted": True
            },
            async_processing=True
        )

        st.info(
            "Completed live lifts are saved to "
            "safelift_reba_records.csv."
        )

        @st.fragment(run_every="2s")
        def live_records_panel():
            processor = webrtc_ctx.video_processor

            # Remember the current session so the results stay visible
            # after the camera is stopped (same as video upload).
            if processor is not None:
                st.session_state["live_session_start"] = (
                    processor.tracker.session_start.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )

            session_start = st.session_state.get("live_session_start")

            if session_start is None:
                st.info(
                    "Press START to begin. The lift count starts "
                    "from 0 every time the camera is started."
                )
                return

            try:
                df = pd.read_csv(CSV_FILE)
            except Exception:
                df = pd.DataFrame(columns=HEADERS)

            df = df[df["Time"].astype(str) >= session_start]

            if df.empty:
                st.info(
                    "No completed lifts in this session yet. Complete a "
                    "full lift cycle in front of the camera."
                )
                return

            st.success(
                f"Detected {len(df)} completed lift(s) in this session "
                f"(started {session_start})."
            )
            render_results(df, prefix="live_lift_results")

        live_records_panel()

    except ImportError:
        st.error(
            "Live camera requires streamlit-webrtc and av. "
            "Install them using:"
        )
        st.code("python -m pip install streamlit-webrtc av")


# ============================================================
# UPLOAD VIDEO MODE
# ============================================================

elif mode == "Upload video":
    st.header("🎬 Analyze a lifting video")

    st.write(
        "Upload a video showing the person's full body. "
        "Video results are displayed separately from saved live-camera records."
    )

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_upload"
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("▶️ Analyze video", type="primary"):
            extension = os.path.splitext(uploaded_video.name)[1]
            temp_path = os.path.join(
                os.getcwd(), "_safelift_uploaded_video" + extension
            )

            with open(temp_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

            cap = cv2.VideoCapture(temp_path)
            tracker = LiftCycleTracker(
                config.copy(), persist=False, start_count=0
            )
            video_rows = []

            frame_total = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            )

            progress = st.progress(
                0.0, text="Analyzing video frames..."
            )

            try:
                with mp_pose.Pose(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                ) as pose:
                    frame_no = 0

                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break

                        frame_no += 1
                        annotated, measurements = process_pose_frame(
                            frame, pose
                        )

                        if measurements:
                            record, stage = tracker.update(
                                annotated, measurements
                            )
                            if record is not None:
                                video_rows.append(record)

                        if frame_total > 0 and frame_no % 10 == 0:
                            progress.progress(
                                min(frame_no / frame_total, 1.0),
                                text=(
                                    f"Analyzing frame "
                                    f"{frame_no}/{frame_total}"
                                )
                            )

                progress.progress(1.0, text="Video analysis complete")

            finally:
                cap.release()
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

            if video_rows:
                st.success(
                    f"Detected {len(video_rows)} completed lift(s)."
                )
                render_results(
                    pd.DataFrame(video_rows),
                    prefix="video_lift_results"
                )
            else:
                st.warning(
                    "No complete lift cycle was detected. Ensure the "
                    "person is fully visible, starts upright, bends, "
                    "and returns upright."
                )


# ============================================================
# UPLOAD CSV MODE
# ============================================================

elif mode == "Upload CSV":
    st.header("📊 Analyze an existing CSV")

    uploaded_csv = st.file_uploader(
        "Upload your notebook or SafeLift CSV",
        type=["csv"],
        key="csv_upload"
    )

    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)

            if df.empty:
                st.info("The uploaded CSV has no rows.")
            else:
                st.success(
                    f"Loaded {len(df)} record(s). "
                    "Recorded scores are preserved."
                )

                # Recognize common alternate notebook column names.
                rename = {}

                candidates = {
                    "Max_Torso_Angle_Deg": [
                        "Torso_Angle_Deg", "Torso °"
                    ],
                    "Min_Knee_Angle_Deg": [
                        "Knee_Angle_Deg", "Knee °"
                    ],
                    "Neck_Angle_Deg": [
                        "Max_Neck_Flexion_Estimate_Deg", "Neck °"
                    ],
                    "Upper_Arm_Angle_Deg": [
                        "Max_Upper_Arm_Angle_Deg", "Arm °"
                    ],
                    "Elbow_Angle_Deg": [
                        "Min_Elbow_Angle_Deg", "Elbow °"
                    ],
                    "Cycle_Number": ["Lift", "Lift_Number"],
                    "Cycle_Duration_Seconds": [
                        "Duration (s)", "Duration_Seconds"
                    ],
                    "REBA_Score": ["REBA"],
                    "REBA_Action_Level": [
                        "Risk level", "Action_Level"
                    ]
                }

                for target, alternatives in candidates.items():
                    if target not in df.columns:
                        for alt in alternatives:
                            if alt in df.columns:
                                rename[alt] = target
                                break

                if rename:
                    df = df.rename(columns=rename)

                render_results(
                    df,
                    prefix="uploaded_csv_results"
                )

        except Exception as e:
            st.error(f"Could not read CSV: {e}")


# ============================================================
# VIEW SAVED LIVE-CAMERA RECORDS
# ============================================================

elif mode == "View lift records":
    st.header("📚 Saved lift records")

    if not os.path.exists(CSV_FILE):
        st.info(
            "No CSV exists yet. Start live-camera detection to create it."
        )
    else:
        df = pd.read_csv(CSV_FILE)

        if df.empty:
            st.info("No lift cycles have been recorded yet.")
        else:
            render_results(
                df,
                prefix="saved_lift_results"
            )

            st.subheader("📊 Risk-level distribution")
            if "REBA_Action_Level" in df.columns:
                st.bar_chart(df["REBA_Action_Level"].value_counts())


st.divider()

st.caption(
    "SafeLift AI 🦺 | Posture-based screening support. "
    "Angle explanations are guidance, not a substitute for a full "
    "ergonomic assessment. Live-camera records are saved to "
    "safelift_reba_records.csv."
)