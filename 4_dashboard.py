# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
from typing import Optional
import streamlit.components.v1 as components
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# XGBoost optional
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# --------------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="리튬 이온 품질관리 대시보드",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------------
# Custom CSS (공통)
# --------------------------------------------------------------------------------
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        margin-bottom: 1rem;
    }
    .metric-value { font-size: 2rem; font-weight: 600; color: #1f2937; }
    .metric-label {
        font-size: 0.875rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em;
    }
    .metric-change { font-size: 0.875rem; margin-top: 0.5rem; }
    .positive { color: #10b981; } .negative { color: #ef4444; }
    div[data-testid="stSidebar"] { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%; background-color: #4f46e5; color: white; border: none;
        padding: 0.6rem 1rem; border-radius: 0.375rem; font-weight: 500;
    }
    .stButton>button:hover { background-color: #4338ca; }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# Initialize session state
# --------------------------------------------------------------------------------
if 'page' not in st.session_state:
    st.session_state.page = 'realtime'   # ✅ 1페이지 = 실시간 대시보드
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'experiment_data' not in st.session_state:
    st.session_state.experiment_data = {}
if 'cycle_data' not in st.session_state:
    st.session_state.cycle_data = None

# ==== Query-Param Navigation (버전 호환) ====
def _read_goto_param():
    # 최신(st.query_params) & 구버전(experimental_get_query_params) 모두 대응
    if hasattr(st, "query_params"):
        qp = st.query_params
        return qp.get("goto", None)
    else:
        params = st.experimental_get_query_params()
        return (params.get("goto", [None]) or [None])[0]

def _clear_query_params():
    # 최신/구버전 모두에서 안전하게 초기화
    if hasattr(st, "query_params"):
        try:
            st.query_params.clear()
        except Exception:
            pass
    else:
        st.experimental_set_query_params()

def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

def handle_nav_query():
    goto = _read_goto_param()
    mapping = {'overview_new': 'realtime', 'realtime': 'realtime', 'prediction': 'prediction', 'main': 'main'}
    if goto in mapping:
        st.session_state.page = mapping[goto]
        _clear_query_params()
        _safe_rerun()

handle_nav_query()

# --------------------------------------------------------------------------------
# Sidebar (항상 표시) — 모델링 버튼을 마지막에 배치
# --------------------------------------------------------------------------------
st.markdown(
     """
    <style>
    .stButton>button {
        width: 200px;             /* 버튼 가로 고정 */
        font-size: 28px;          /* 버튼 내부 글자 크기 */
        text-align: center;   /* 버튼 내부 글자 가운데 정렬 */
        padding: 6px 10px;
        display: block;       /* block으로 만들어 margin 적용 가능 */
        margin-left: auto;    /* 좌우 자동 여백 */
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("### 바로가기")
    if st.button("실시간 대시보드", key="nav_realtime"):
        st.session_state.page = 'realtime'; _safe_rerun()
    if st.button("메인 대시보드", key="nav_main"):
        st.session_state.page = 'main'; _safe_rerun()
    if st.button("모델링 예측", key="nav_prediction"):
        st.session_state.page = 'prediction'; _safe_rerun()
    st.markdown("---")

# --------------------------------------------------------------------------------
# Battery ID groups 
BATTERY_GROUPS = {
    'Group 1 (B5-7,18, 24°C)': ['B0005', 'B0006', 'B0007', 'B0018'],
    'Group 2 (B25-28, 24°C)': ['B0025', 'B0026', 'B0027', 'B0028'],
    'Group 3 (B29-32, 43°C)': ['B0029', 'B0030', 'B0031', 'B0032'],
    'Group 4 (B33-34,36, 24°C)': ['B0033', 'B0034', 'B0036'],
    'Group 5 (B38-40, Multi-temp)': ['B0038', 'B0039', 'B0040'],
    'Group 6 (B41-44, 4°C)': ['B0041', 'B0042', 'B0043', 'B0044'],
    'Group 7 (B45-48, 4°C)': ['B0045', 'B0046', 'B0047', 'B0048'],
    'Group 8 (B49-52, 4°C)': ['B0049', 'B0050', 'B0051', 'B0052'],
    'Group 9 (B53-56, 4°C)': ['B0053', 'B0054', 'B0055', 'B0056']
}

# --------------------------------------------------------------------------------
# 실시간 처리 유틸
# --------------------------------------------------------------------------------
def delete_dot(row):
    answer = []
    for i in row[:-1]:
        num = str(i).replace('.', '')
        answer.append(num)
    answer.append(str(row[-1]))
    return answer

def as_int(row):
    answer = []
    for i in range(len(row)):
        if i == len(row) - 1:
            num = float(row[i])
            s = str(num)
            answer.append(float(s))
        else:
            answer.append(int(float(row[i])))
    return answer

def process_row(row):
    for i in row:
        if 'e' in i:
            return as_int(row)
    else:
        return delete_dot(row)

def datetime_change(row):
    year = str(row[0]); month = str(row[1]); day = str(row[2])
    hour = str(row[3]); minute = str(row[4]); val = float(row[5])
    if val >= 1000:
        sec = str(int(val // 1000)).zfill(2)
        msec = str(int(val % 1000)).zfill(3)
    else:
        sec = str(int(val)).zfill(2)
        msec = str(int(round((val - int(val)) * 1000))).zfill(3)
    timestamp = f"{year}-{month}-{day} {hour}:{minute}:{sec}.{msec}"
    return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")

# 안전 포맷/변환 유틸 (테이블/경계 체크용)
def _fmt(v, nd=3):
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)

def _to_float(v):
    try:
        return float(v)
    except Exception:
        return None

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Page 3: Predictive Modeling
# --------------------------------------------------------------------------------
# Page 3: Predictive Modeling
# --------------------------------------------------------------------------------
def run_predictive_modeling():
    import streamlit as st
    import streamlit.components.v1 as components

    st.title("예측 모델링 - RUL 예측")

    # XGBoost availability guard
    try:
        from xgboost import XGBRegressor
        _xgb_ok = True
    except Exception:
        _xgb_ok = False

    # -------------------------- Helpers --------------------------
    def clamp(v, lo, hi):
        try:
            v = float(v)
        except Exception:
            return lo
        return max(lo, min(hi, v))

    def color_for_percent(p):
        if p <= 33:
            return "#ef4444"  # red
        elif p <= 66:
            return "#f59e0b"  # amber
        return "#22c55e"      # green

    # 넓은 가로폭 + 단자 잘림 방지 + 모드(채움/방전) + 우측 표시 텍스트 커스텀
    def render_battery(
        percent: float,
        title: str = "",
        subtitle: str = "",
        height: int = 260,
        min_width: int = 700,
        max_width: int = 1200,
        mode: str = "fill",
        right_value_text: Optional[str] = None,
        right_caption: str = "현재 값"
    ):
        """
        percent: 0~100
        mode:
          - "fill"  : 0%에서 목표까지 채워짐 (SoH)
          - "drain" : 100%에서 목표까지 줄어듦 (RUL)
        """
        percent_ = clamp(percent, 0, 100)
        col = color_for_percent(percent_)

        # 단자 크기 스케일
        term_w = max(24, int(height * 0.12))
        term_h = max(54, int(height * 0.24))
        term_r = max(8,  int(height * 0.036))

        anim = "fillAnim" if mode == "fill" else "drainAnim"
        start_w = "0%" if mode == "fill" else "100%"
        width_css = f"clamp({min_width}px, 100%, {max_width}px)"

        right_val = f"{percent_:.1f}%" if right_value_text is None else right_value_text

        html = f"""
        <div class="battery-wrap" style="display:flex; align-items:center; gap:28px; width:100%;">
          <div class="battery-left" style="flex:1 1 auto; min-width:0;">
            <div style="font-weight:700; font-size:22px; margin-bottom:8px; white-space:normal;">{title}</div>
            <div style="position:relative; width:{width_css}; padding-right:{term_w + 12}px; box-sizing:border-box; overflow:visible;">
              <div style="position:relative; width:100%; min-width:{min_width}px; height:{height}px; border:6px solid #1f2937; border-radius:18px; background:linear-gradient(#fff,#f7f7f7); box-shadow:0 6px 18px rgba(0,0,0,.08); overflow:hidden;">
                <div style="--target:{percent_}%; background:{col}; position:absolute; left:0; top:0; bottom:0; width:{start_w}; margin:6px; border-radius:12px; animation:{anim} 1.2s ease-out forwards; background-size:20px 20px; background-image:linear-gradient(45deg, rgba(255,255,255,.25) 25%, transparent 25%, transparent 50%, rgba(255,255,255,.25) 50%, rgba(255,255,255,.25) 75%, transparent 75%, transparent);"></div>
                <div style="position:absolute; inset:6px; border-radius:12px; box-shadow:inset 0 0 0 1px rgba(0,0,0,.06)"></div>
              </div>
              <div style="position:absolute; right:0; top:calc(50% - {term_h/2}px); width:{term_w}px; height:{term_h}px; background:#1f2937; border-radius:{term_r}px;"></div>
            </div>
            <div style="color:#6b7280; margin-top:8px; font-size:16px;">{subtitle}</div>
          </div>
          <div style="flex:0 0 200px; text-align:center;">
            <div style="font-size:18px; color:#6b7280; margin-bottom:6px;">{right_caption}</div>
            <div style="font-size:52px; font-weight:800; line-height:1; color:{col}">{right_val}</div>
          </div>
        </div>
        <style>
          @keyframes fillAnim  {{ from {{ width:0%;   }} to {{ width:var(--target); }} }}
          @keyframes drainAnim {{ from {{ width:100%; }} to {{ width:var(--target); }} }}
        </style>
        """
        return html

    # -------------------------- Tabs --------------------------
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 28px;
        font-weight: 800;       
    }
    /* Streamlit 전체 글자 크기 증가 */
    .stSelectbox label, .stMultiSelect label, .stNumberInput label, .stFileUploader label {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    .stRadio label, .stSlider label {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    .stExpander summary {
        font-size: 20px !important;
        font-weight: 700 !important;
    }
    .stButton button {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    .stInfo, .stWarning, .stError, .stSuccess {
        font-size: 16px !important;
    }
    .stMarkdown p, .stMarkdown li {
        font-size: 16px !important;
    }
    .stMarkdown h4 {
        font-size: 22px !important;
        font-weight: 700 !important;
    }
    .stMarkdown h3 {
        font-size: 26px !important;
        font-weight: 700 !important;
    }
    .stDataFrame {
        font-size: 14px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["데이터·학습", "예측"])

    # ===================== Tab 1: Data & Training =====================
    with tabs[0]:
        st.markdown("### 학습 데이터 업로드")
        uploaded_file = st.file_uploader(
            "학습용 CSV 파일 선택",
            type="csv",
            help="RUL 또는 SoH 등을 포함한 학습 데이터셋을 업로드하세요.",
            key="mdl_train_csv"
        )

        df = None
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            # 데이터셋 정보와 모델 학습을 2칸으로 나누어 배치
            left_col, right_col = st.columns([1, 1])

            # 왼쪽: 데이터셋 정보
            with right_col:
                with st.expander("데이터셋 정보", expanded=True):
                    st.markdown("### 데이터셋 정보")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.metric("샘플 수", f"{len(df):,}")
                    with col2:
                        st.metric("특성 수", f"{len(df.columns)}")
                    
                    with st.expander("미리보기"):
                        # Unnamed:0 컬럼 제거
                        display_df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                        st.dataframe(display_df.head(10), use_container_width=True)
                        st.markdown("#### 기초 통계")
                        st.dataframe(display_df.describe(include='all'), use_container_width=True)

            # 오른쪽: 모델 학습 설정
            with left_col:
                with st.expander("모델 설정", expanded=True):
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

                    default_input_features = [
                        "rest_time", "oven_time_cha", "oven_time_disch",
                        "ambient_temperature", "current_load", "cutoff_voltage",
                        "used_time", "Re", "Rct", "total_R"
                    ]
                    available_input_features = [f for f in default_input_features if f in df.columns]

                    selected_features = st.multiselect(
                        "입력 특성",
                        options=numeric_cols,
                        default=available_input_features if available_input_features else numeric_cols[:10],
                        help="예측에 사용할 입력 특성을 선택하세요.",
                        key="mdl_feat_sel"
                    )

                    target_options = ["RUL", "corr_capacity2", "SoH2"]
                    available_targets = [t for t in target_options if t in df.columns]
                    if not available_targets:
                        available_targets = [col for col in numeric_cols if any(
                            kw in col.lower() for kw in ['rul', 'capacity', 'soh']
                        )] or numeric_cols[:3]

                    # 단일 예측 탭에서 전환하기 위해 기억
                    st.session_state['available_targets'] = available_targets
                    
                    def to_display_name(name: str) -> str:
                        m = {
                            "rul": "RUL",
                            "corr_capacity2": "Capacity",
                            "capacity": "Capacity",
                            "soh2": "SoH",
                            "soh": "SoH",
                            "soh%": "SoH",
                        }
                        return m.get(str(name).lower(), str(name))

                    target_variable = st.selectbox(
                        "목표 변수 선택",
                        options=available_targets,
                        index=0,
                        help="예측할 목표 변수를 선택하세요.",
                        format_func=to_display_name,
                        key="mdl_target"
                    )

                    with st.expander("상세 설정", expanded=False):
                        st.markdown("**모델링 파라미터**")
                        test_size = st.slider("검증 데이터 비율(%)", 10, 40, 20, key="mdl_test_size") / 100
                        random_seed = st.number_input("랜덤 시드", 0, 100, 42, key="mdl_seed")

                        st.markdown("**XGBoost 하이퍼파라미터**")
                        st.info("고정된 최적값을 사용합니다.")
                        st.markdown("""
                        • **N Estimators:** 500  
                        • **Max Depth:** 5  
                        • **Learning Rate:** 0.1  
                        • **Colsample Bytree:** 1.0  
                        • **Reg Lambda:** 5.0  
                        • **Subsample:** 0.8
                        """)

                    st.markdown("---")
                    if st.button("모델 학습", type="primary", use_container_width=True, key="mdl_train_btn"):
                        if not _xgb_ok:
                            st.error("XGBoost가 설치되어 있지 않습니다. `pip install xgboost` 후 다시 시도하세요.")
                        elif target_variable not in df.columns:
                            st.error(f"'{target_variable}' 컬럼을 찾을 수 없습니다.")
                        elif not selected_features:
                            st.error("적어도 하나의 입력 특성을 선택하세요.")
                        else:
                            with st.spinner("모델을 학습 중입니다."):
                                X = df[selected_features].copy()
                                y = df[target_variable].values

                                # RUL 기준값(배터리 애니메이션 스케일링용)
                                try:
                                    y_series = pd.Series(y).dropna().astype(float)
                                    st.session_state['target_stats'] = {
                                        'min': float(y_series.min()),
                                        'max': float(y_series.max()),
                                        'p95': float(y_series.quantile(0.95)),
                                        'mean': float(y_series.mean()),
                                        'median': float(y_series.median())
                                    }
                                except Exception:
                                    st.session_state['target_stats'] = None

                                imputer = SimpleImputer(strategy='mean')
                                X_imputed = imputer.fit_transform(X)

                                X_train, X_test, y_train, y_test = train_test_split(
                                    X_imputed, y, test_size=test_size, random_state=random_seed
                                )

                                scaler = StandardScaler()
                                X_train_scaled = scaler.fit_transform(X_train)
                                X_test_scaled = scaler.transform(X_test)

                                model = XGBRegressor(
                                    n_estimators=500,
                                    max_depth=5,
                                    learning_rate=0.1,
                                    colsample_bytree=1.0,
                                    reg_lambda=5.0,
                                    subsample=0.8,
                                    objective="reg:squarederror",
                                    random_state=random_seed,
                                    n_jobs=-1,
                                    tree_method="hist"
                                )
                                model.fit(X_train_scaled, y_train)

                                y_pred = model.predict(X_test_scaled)
                                mse = mean_squared_error(y_test, y_pred)
                                rmse = float(np.sqrt(mse))
                                mae = mean_absolute_error(y_test, y_pred)
                                r2 = r2_score(y_test, y_pred)

                                y_train_pred = model.predict(X_train_scaled)
                                train_r2 = r2_score(y_train, y_train_pred)
                                overfitting_indicator = float(train_r2 - r2)

                                # Save to session
                                st.session_state['model'] = model
                                st.session_state['scaler'] = scaler
                                st.session_state['imputer'] = imputer
                                st.session_state['selected_features'] = selected_features
                                st.session_state['target_variable'] = target_variable
                                st.session_state['y_test'] = y_test
                                st.session_state['y_pred'] = y_pred
                                st.session_state['metrics'] = {
                                    'mse': float(mse), 'rmse': rmse, 'mae': float(mae),
                                    'r2': float(r2), 'overfitting': overfitting_indicator
                                }
                                st.session_state['feature_importances'] = model.feature_importances_

                                # 단일 예측에서 '빠른 재학습'에 사용 (✅ 위젯 키와 다른 이름 사용)
                                st.session_state['last_train_df'] = df
                                st.session_state['last_test_size'] = float(test_size)   # ✅
                                st.session_state['last_seed'] = int(random_seed)         # ✅

                                st.success("모델 학습이 완료되었습니다.")

            # 모델링 결과를 대시보드 스타일로 배치
            if 'model' in st.session_state:
                st.markdown(
                    "<h2 style='font-size:30px; font-weight:800; color:black; margin:0 0 8px 0;'>모델링 결과</h2>",
                    unsafe_allow_html=True
                )

                y_test = st.session_state['y_test']
                y_pred = st.session_state['y_pred']
                metrics = st.session_state['metrics']
                target_var = st.session_state.get('target_variable', 'Target')
                
                disp_target = to_display_name(target_var)

                # 1x3 레이아웃: 실측값 vs 예측값 / 변수 중요도 / 성능 지표들
                viz_col1, viz_col2, metrics_col = st.columns([1, 1, 1])

                # 첫 번째: 실측값 vs 예측값 그래프
                with viz_col1:
                    st.markdown("<div style='height:80px;'></div>", unsafe_allow_html=True)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=y_test, y=y_pred, mode='markers', name='예측값',
                        marker=dict(size=8, opacity=0.6),
                        text=[f"실측: {a:.3f}<br>예측: {p:.3f}<br>오차: {abs(a-p):.3f}"
                              for a, p in zip(y_test, y_pred)],
                        hovertemplate="%{text}<extra></extra>"
                    ))
                    min_val = float(min(np.min(y_test), np.min(y_pred)))
                    max_val = float(max(np.max(y_test), np.max(y_pred)))
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                        name='완전 일치선', line=dict(dash='dash', width=2)
                    ))
                    if len(y_test) >= 2:
                        z = np.polyfit(y_test, y_pred, 1)
                        pfit = np.poly1d(z)
                        x_trend = np.linspace(min_val, max_val, 100)
                        fig.add_trace(go.Scatter(
                            x=x_trend, y=pfit(x_trend), mode='lines',
                            name='추세선', line=dict(width=2)
                        ))
                    fig.update_layout(
                        title=dict(text=f"실측값 vs 예측값 ({disp_target})", font=dict(size=32, color='black')),
                        xaxis=dict(
                            title=dict(text=f"실측 {disp_target}", font=dict(size=20, color='black')),
                            tickfont=dict(size=16, color='black')
                        ),
                        yaxis=dict(
                            title=dict(text=f"예측 {disp_target}", font=dict(size=20, color='black')),
                            tickfont=dict(size=16, color='black')
                        ),
                        legend=dict(font=dict(size=16, color='black')),
                        height=600, hovermode='closest', showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # 두 번째: 변수 중요도 시각화
                with viz_col2:
                    features = st.session_state['selected_features']
                    importances = st.session_state['feature_importances']
                    fi_df = pd.DataFrame({'변수': features, '중요도': importances})
                    max_importance = float(fi_df['중요도'].max()) if not fi_df.empty else 0.0

                    importance_type = st.radio(
                        "중요도 보기:", ["상위", "하위"],
                        horizontal=True, key="fi_pick"
                    )
                    if importance_type == "상위":
                        fi_plot = fi_df.nlargest(5, '중요도').sort_values('중요도', ascending=False)
                        title = "상위 5개 변수"
                    else:
                        fi_plot = fi_df.nsmallest(5, '중요도').sort_values('중요도', ascending=True)
                        title = "하위 5개 변수"

                    fig_fi = go.Figure()
                    fig_fi.add_trace(go.Bar(
                        y=fi_plot['변수'][::-1],
                        x=fi_plot['중요도'][::-1],
                        orientation='h',
                        hovertemplate='변수: %{y}<br>중요도: %{x:.4f}<extra></extra>'  # 마우스 오버 전용
                    ))
                    x_range = [0, max_importance] if max_importance > 0 else None
                    fig_fi.update_layout(
                    title=dict(text=title, font=dict(size=32, color='black')),
                    xaxis=dict(
                        title=dict(text="중요도 점수", font=dict(size=20, color='black')),
                        tickfont=dict(size=14, color='black')
                    ),
                    yaxis=dict(
                        title=dict(text="변수", font=dict(size=20, color='black')),
                        tickfont=dict(size=16, color='black')
                    ),
                    legend=dict(font=dict(size=16, color='black')),
                    height=600, margin=dict(l=200),
                    xaxis_range=x_range if x_range else None
                )
                    st.plotly_chart(fig_fi, use_container_width=True)

                # 세 번째: 성능 지표들을 큰 박스 안에 2x2로 배치
                with metrics_col:
                    st.markdown("""
                        <h2 style="font-size:32px; font-weight:bold; color:black; margin-bottom:10px;">
                            모델 성능 지표
                        </h2>
                    """, unsafe_allow_html=True)
                    card_style = """
                        <div style="
                            background-color: #f8f9fa;
                            padding: 20px;
                            border-radius: 10px;
                            text-align: center;
                            border: 1px solid #ddd;
                            height: 280px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;
                            margin: 10px 10px;
                        ">
                            <h3 style="margin: 0; font-size: 56px; color: #000;">{title}</h3>
                            <h2 style="margin: 5px 0 0 0; font-size: 42px; color: #000;">{value}</h2>
                        </div>
                    """
                    metric_row1_col1, metric_row1_col2 = st.columns(2)
                    with metric_row1_col1:
                        st.markdown(card_style.format(title="R²", value=f"{metrics['r2']:.4f}"), unsafe_allow_html=True)
                    with metric_row1_col2:
                        st.markdown(card_style.format(title="MAE", value=f"{metrics['mae']:.3f}"), unsafe_allow_html=True)

                    metric_row2_col1, metric_row2_col2 = st.columns(2)
                    with metric_row2_col1:
                        st.markdown(card_style.format(title="MSE", value=f"{metrics['mse']:.3f}"), unsafe_allow_html=True)
                    with metric_row2_col2:
                        st.markdown(card_style.format(title="RMSE", value=f"{metrics['rmse']:.3f}"), unsafe_allow_html=True)

            else:
                st.info("먼저 모델을 학습하세요.")

        if df is None:
            st.info("CSV 파일을 업로드해 시작하세요.")
    # ===================== Tab 2: Predictions =====================
    with tabs[1]:
        st.markdown("### 새 예측")

        if 'model' not in st.session_state:
            st.info("먼저 '데이터·학습' 탭에서 모델을 학습하세요.")
            return

        # --- 라디오: 문자열 비교 대신 인덱스 기반으로 안전 분기 ---
        METHODS = ["CSV 파일로 일괄 예측", "수동 입력으로 단일 예측"]
        st.session_state.setdefault("pred_method_idx", 0)
        method_idx = st.radio(
            "예측 방식 선택",
            options=[0, 1],
            format_func=lambda i: METHODS[i],
            key="pred_method_idx"
        )

        # -------- Batch prediction ----------
        if method_idx == 0:
            pred_file = st.file_uploader("예측용 CSV 업로드", type="csv", key="prediction_file")
            if pred_file is not None:
                pred_df = pd.read_csv(pred_file)
                required_features = st.session_state['selected_features']
                missing_features = [f for f in required_features if f not in pred_df.columns]

                if missing_features:
                    st.error(f"누락된 특성: {missing_features}")
                else:
                    if st.button("예측 실행", key="batch_predict"):
                        X_new = pred_df[required_features]
                        X_new_imputed = st.session_state['imputer'].transform(X_new)
                        X_new_scaled = st.session_state['scaler'].transform(X_new_imputed)
                        predictions = st.session_state['model'].predict(X_new_scaled)

                        target_var = st.session_state['target_variable']
                        pred_df[f'Predicted_{target_var}'] = predictions

                        st.session_state['pred_results'] = pred_df
                        st.session_state['target_var'] = target_var

                        st.success("예측이 완료되었습니다.")

            if 'pred_results' in st.session_state:
                pred_df = st.session_state['pred_results'].copy()
                target_var = st.session_state['target_var']

                # ======================= 좌측: 상태 카드 UI =======================
                left_col, right_col = st.columns([1, 1])

                # ======================= 좌측: 상태 카드 + 파이차트 =======================
                with left_col:
                    if 'battery_id' not in pred_df.columns:
                        st.warning("'battery_id' 컬럼이 데이터에 없습니다.")
                    else:
                        # 각 배터리의 "마지막" 예측 RUL
                        last_predictions = (
                            pred_df
                            .sort_values(['battery_id', 'cycle'])  # 안전 정렬
                            .groupby('battery_id')[f'Predicted_{target_var}']
                            .last()
                        )

                        # 카테고리 분류
                        def classify(v):
                            try:
                                v = float(v)
                            except:
                                return "기타"
                            if v <= 50: return "위험"
                            if v <= 100: return "주의"
                            return "양호"

                        cats = {"위험": [], "주의": [], "양호": []}
                        for bid, val in last_predictions.items():
                            c = classify(val)
                            if c in cats:
                                cats[c].append((bid, float(val)))

                        # ---- 파이차트 (RUL 구간별 분포) ----
                        order = ["위험", "주의", "양호"]
                        labels = [f"{k} ({len(cats[k])}개)" for k in order]
                        values = [len(cats[k]) for k in order]
                        colors = ["#ef4444", "#f59e0b", "#10b981"]

                        fig_pie = go.Figure(
                            data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors))]
                        )
                        fig_pie.update_layout(
                        title=dict(
                            text="RUL 구간별 분포",
                            font=dict(size=24, color="black"),  # 글자 크기와 색상
                        ),
                        height=320,
                        margin=dict(t=40, b=0, l=0, r=0)
)
                        st.plotly_chart(fig_pie, use_container_width=True)

                        # 카드용 CSS
                        st.markdown("""
                        <style>
                        .card-wrap{display:block; border-radius:12px; padding:14px 16px; border:1px solid #e5e7eb;}
                        .card-danger{background:#fdecef;}
                        .card-warn{background:#fff7e6;}
                        .card-good{background:#e7f6ee;}
                        .dot{display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; position:relative; top:-1px;}
                        .dot-red{background:#ef4444;} .dot-yellow{background:#f59e0b;} .dot-green{background:#10b981;}
                        .subtle{color:#6b7280; font-size:18px;}
                        .item{font-size:19px; margin:6px 0;}
                        </style>
                        """, unsafe_allow_html=True)

                        c1, c2, c3 = st.columns(3)

                        def render_card(col, title, items, cls, dot_cls, guide):
                            with col:
                                col.markdown(
                                    f"""<div class="card-wrap {cls}">
                                    <div style="font-weight:700; font-size:22px;">
                                    <span class="dot {dot_cls}"></span>{title} ({len(items)}개)
                                    </div>
                                    <div class="subtle" style="margin-top:6px;">{guide}</div>
                                    <div style="margin-top:10px; max-height:220px; overflow:auto;">
                                    {"".join([f'<div class="item">• {bid}: {val:.0f} cycles</div>' for bid,val in items]) or "<div class='item subtle'>해당 없음</div>"}
                                    </div>
                                    </div>""",
                                    unsafe_allow_html=True
                                )

                        render_card(c1, "위험", cats["위험"], "card-danger", "dot-red", "RUL ≤ 50 cycles")
                        render_card(c2, "주의", cats["주의"], "card-warn", "dot-yellow", "50 < RUL ≤ 100 cycles")
                        render_card(c3, "양호", cats["양호"], "card-good", "dot-green", "RUL > 100 cycles")


                # ======================= 우측: 개별(다중) 분석 =======================
                with right_col:
                    st.markdown("#### 개별 배터리 분석 (다중 선택)")

                    if 'battery_id' not in pred_df.columns:
                        st.warning("'battery_id' 컬럼이 데이터에 없습니다.")
                    else:
                        unique_batteries = pred_df['battery_id'].unique().tolist()
                        default_sel = unique_batteries[:3] if unique_batteries else []
                        selected_batteries = st.multiselect(
                            "비교할 배터리 선택",
                            options=unique_batteries,
                            default=default_sel,
                            key="pred_batt_multi_new"
                        )

                        if not selected_batteries:
                            st.info("위에서 비교할 배터리를 선택하세요.")
                        else:
                            # ---------- 1) 사이클 vs RUL (다중 선택 + 우측 요약 패널) ----------
                            st.markdown("##### RUL vs 사이클")
                            plot_col, panel_col = st.columns([3.0, 1.35])  # ▶︎ 우측 패널 포함 레이아웃

                            with plot_col:
                                fig_rul = go.Figure()
                                for bid in selected_batteries:
                                    bdat = pred_df[pred_df['battery_id'] == bid].sort_values('cycle')
                                    if f'Predicted_{target_var}' in bdat.columns and 'cycle' in bdat.columns:
                                        fig_rul.add_trace(go.Scatter(
                                            x=bdat['cycle'], y=bdat[f'Predicted_{target_var}'],
                                            mode='lines+markers', name=f'{bid} (예측)'
                                        ))
                                    if 'RUL' in bdat.columns and 'cycle' in bdat.columns:
                                        fig_rul.add_trace(go.Scatter(
                                            x=bdat['cycle'], y=bdat['RUL'],
                                            mode='lines+markers', name=f'{bid} (실측)', line=dict(dash='dot')
                                        ))

                                # 축 제목과 눈금 폰트 크기, 색상 조정
                                fig_rul.update_layout(
                                    xaxis=dict(
                                        title=dict(text="Cycle", font=dict(size=18, color="black")),
                                        tickfont=dict(size=14, color="black")
                                    ),
                                    yaxis=dict(
                                        title=dict(text="RUL (cycles)", font=dict(size=18, color="black")),
                                        tickfont=dict(size=14, color="black")
                                    ),
                                    height=480,
                                    legend=dict(font=dict(size=16, color="black"))
                                )

                                st.plotly_chart(fig_rul, use_container_width=True)

                            with panel_col:
                                st.markdown("###### ")  # 공간
                                # 드롭다운을 패널 오른쪽 상단처럼 배치
                                pad1, pad2 = st.columns([1, 1])
                                with pad2:
                                    picked_rul = st.selectbox(
                                        " ", options=selected_batteries, key="rul_pick_panel", label_visibility="collapsed"
                                    )

                                # 값 계산
                                b0 = pred_df[pred_df['battery_id'] == picked_rul].sort_values('cycle')
                                cur_rul = float(b0[f'Predicted_{target_var}'].iloc[-1]) if f'Predicted_{target_var}' in b0 else None
                                init_rul = float(b0[f'Predicted_{target_var}'].iloc[0]) if f'Predicted_{target_var}' in b0 else None

                                st.markdown(
                                    f"""
                                    <div style="background:#f3f4f6; border:1px dashed #d1d5db; border-radius:12px; padding:18px; height:360px; display:flex; flex-direction:column; justify-content:center; align-items:center;">
                                    <div style="font-size:24px; color:#111827; margin-bottom:8px;">현재 RUL</div>
                                    <div style="font-size:54px; font-weight:800; color:#111827; line-height:1;">{(cur_rul if cur_rul is not None else float('nan')):.1f} cycles</div>
                                    <div style="margin-top:10px; font-size:26px; color:#dc2626;">초기 RUL: {(init_rul if init_rul is not None else float('nan')):.1f} cycles</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            st.markdown("---")

                            # ---------- 2) 사이클 vs Total R (다중 선택 + 우측 요약 패널) ----------
                            st.markdown("##### Total R vs 사이클")
                            plot_col2, panel_col2 = st.columns([3.0, 1.35])

                            with plot_col2:
                                fig_tr = go.Figure()
                                for bid in selected_batteries:
                                    bdat = pred_df[pred_df['battery_id'] == bid].sort_values('cycle')
                                    if 'total_R' in bdat.columns and 'cycle' in bdat.columns:
                                        fig_tr.add_trace(go.Scatter(
                                            x=bdat['cycle'], y=bdat['total_R'],
                                            mode='lines+markers', name=f'{bid} (Total R)'
                                        ))

                                # 축 제목과 눈금 폰트 크기, 색상 조정
                                fig_tr.update_layout(
                                    xaxis=dict(
                                        title=dict(text="Cycle", font=dict(size=18, color="black")),
                                        tickfont=dict(size=14, color="black")
                                    ),
                                    yaxis=dict(
                                        title=dict(text="Total R (Ω)", font=dict(size=18, color="black")),
                                        tickfont=dict(size=14, color="black")
                                    ),
                                    legend=dict(
                                        font=dict(size=16, color="black")
                                    ),
                                    height=480
                                )

                                st.plotly_chart(fig_tr, use_container_width=True)

                            with panel_col2:
                                st.markdown("###### ")
                                pad3, pad4 = st.columns([1, 1])
                                with pad4:
                                    picked_tr = st.selectbox(
                                        "  ", options=selected_batteries, key="tr_pick_panel", label_visibility="collapsed"
                                    )
                                b1 = pred_df[pred_df['battery_id'] == picked_tr].sort_values('cycle')
                                cur_tr = float(b1['total_R'].iloc[-1]) if 'total_R' in b1 else None
                                init_tr = float(b1['total_R'].iloc[0]) if 'total_R' in b1 else None

                                st.markdown(
                                    f"""
                                    <div style="background:#f3f4f6; border:1px dashed #d1d5db; border-radius:12px; padding:18px; height:360px; display:flex; flex-direction:column; justify-content:center; align-items:center;">
                                    <div style="font-size:24px; color:#111827; margin-bottom:8px;">현재 Total R</div>
                                    <div style="font-size:54px; font-weight:800; color:#111827; line-height:1;">{(cur_tr if cur_tr is not None else float('nan')):.3f} Ω</div>
                                    <div style="margin-top:10px; font-size:26px; color:#dc2626;">초기 Total R: {(init_tr if init_tr is not None else float('nan')):.3f} Ω</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                # ----------------- 결과 CSV 다운로드 -----------------
                st.markdown("---")
                csv = pred_df.to_csv(index=False)
                st.download_button(
                    "예측 결과 CSV 다운로드", data=csv,
                    file_name=f"{target_var.lower()}_predictions.csv", mime="text/csv",
                    key="pred_dl_btn"
                )
        # -------- Single prediction ----------
        else:
            st.markdown("#### 수동 입력으로 단일 예측")

            # 학습 때 저장해둔 정보 로드
            last_df = st.session_state.get('last_train_df')
            selected_features = st.session_state.get('selected_features', [])
            trained_target = st.session_state.get('target_variable', None)

            # 단일예측에서도 종속변수(타깃) 전환
            target_choices = st.session_state.get('available_targets')
            if not target_choices:
                if last_df is not None:
                    num_cols = last_df.select_dtypes(include=[np.number]).columns.tolist()
                    fallback = ["RUL", "corr_capacity2", "SoH2"]
                    target_choices = [c for c in fallback if c in num_cols] or num_cols[:3]
                else:
                    target_choices = ["RUL", "corr_capacity2", "SoH2"]

            try:
                default_idx = target_choices.index(trained_target) if trained_target else 0
            except ValueError:
                default_idx = 0

            # 5칸으로 나누기
            col1, col2, col3, col4, col5 = st.columns(5)
            
            # 표시명 매핑
            display_map = {
                "RUL": "RUL",
                "corr_capacity2": "Capacity",
                "Capacity": "Capacity",
                "SoH2": "SoH",
                "SoH": "SoH",
                "SOH": "SoH",
            }

            # 첫 번째 칸에 selectbox 배치
            with col1:
                pred_target = st.selectbox(
                    "종속변수(목표) 선택",
                    options=target_choices,
                    index=default_idx,
                    key="single_pred_target",
                    format_func=lambda x: display_map.get(x, x),  # 화면 표시만 변경
                    help="여기서 목표(RUL/Capacity/SoH 등)를 바꿀 수 있어요. 예측 시 모델이 자동으로 해당 목표로 재학습됩니다."
                )

            # 나머지 4칸은 빈 칸으로 유지
            with col2:
                st.empty()
            with col3:
                st.empty()
            with col4:
                st.empty()
            with col5:
                st.empty()

            if not selected_features:
                st.info("입력 특성이 없습니다. 먼저 '데이터·학습' 탭에서 모델을 학습하세요.")
                st.stop()

            # 입력값: 접기/펼치기 + 학습데이터 중앙값 기본 채움
            defaults = {}
            if last_df is not None:
                try:
                    defaults = last_df[selected_features].median(numeric_only=True).to_dict()
                except Exception:
                    defaults = {}

            with st.expander("예측에 사용할 입력값", expanded=False):
                cols = st.columns(3)
                for i, feat in enumerate(selected_features):
                    with cols[i % 3]:
                        st.number_input(
                            feat,
                            value=float(defaults.get(feat, 0.0)),
                            format="%.4f",
                            key=f"single_pred_in_{feat}"
                        )

            # 내부 재학습 함수
            def _retrain_for_target(target_name: str):
                if last_df is None:
                    st.error("학습 데이터가 없습니다. '데이터·학습' 탭에서 먼저 모델을 학습해주세요.")
                    return False

                miss = [c for c in selected_features if c not in last_df.columns]
                if miss:
                    st.error(f"학습 데이터에 다음 특성이 없습니다: {miss}")
                    return False
                if target_name not in last_df.columns:
                    st.error(f"학습 데이터에 '{target_name}' 컬럼이 없습니다.")
                    return False

                # ✅ 위젯 키 대신 별도 키 사용
                test_size = float(st.session_state.get("last_test_size", 0.2))  # ✅
                random_seed = int(st.session_state.get("last_seed", 42))        # ✅

                X = last_df[selected_features].copy()
                y = last_df[target_name].values

                # 기준 통계(배터리 애니메이션용)
                try:
                    y_series = pd.Series(y).dropna().astype(float)
                    st.session_state['target_stats'] = {
                        'min': float(y_series.min()),
                        'max': float(y_series.max()),
                        'p95': float(y_series.quantile(0.95)),
                        'mean': float(y_series.mean()),
                        'median': float(y_series.median())
                    }
                except Exception:
                    st.session_state['target_stats'] = None

                imputer = SimpleImputer(strategy='mean')
                X_imp = imputer.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_imp, y, test_size=test_size, random_state=random_seed
                )
                scaler = StandardScaler()
                X_train_sc = scaler.fit_transform(X_train)
                X_test_sc = scaler.transform(X_test)

                if not _xgb_ok:
                    st.error("XGBoost가 설치되어 있지 않습니다. `pip install xgboost` 후 다시 시도하세요.")
                    return False

                model = XGBRegressor(
                    n_estimators=500, max_depth=5, learning_rate=0.1,
                    colsample_bytree=1.0, reg_lambda=5.0, subsample=0.8,
                    objective="reg:squarederror", random_state=random_seed,
                    n_jobs=-1, tree_method="hist"
                )
                with st.spinner(f"'{target_name}' 기준으로 재학습 중..."):
                    model.fit(X_train_sc, y_train)

                # 지표 저장
                y_pred = model.predict(X_test_sc)
                mse = mean_squared_error(y_test, y_pred)
                rmse = float(np.sqrt(mse))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                train_r2 = r2_score(y_train, model.predict(X_train_sc))
                overfit = float(train_r2 - r2)

                # 세션 업데이트
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['imputer'] = imputer
                st.session_state['target_variable'] = target_name
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
                st.session_state['metrics'] = {
                    'mse': float(mse), 'rmse': rmse, 'mae': float(mae),
                    'r2': float(r2), 'overfitting': overfit
                }
                st.session_state['feature_importances'] = model.feature_importances_
                return True

            # 실행 버튼
            if st.button("예측 실행", key="single_predict_btn"):

                # 목표가 바뀌었으면 자동 재학습
                if st.session_state.get('target_variable') != pred_target:
                    ok = _retrain_for_target(pred_target)
                    if not ok:
                        st.stop()

                # 입력 데이터 준비
                X_single = pd.DataFrame([{f: st.session_state.get(f"single_pred_in_{f}", 0.0)
                                        for f in selected_features}])
                X_single_imp = st.session_state['imputer'].transform(X_single)
                X_single_sc = st.session_state['scaler'].transform(X_single_imp)

                # 예측
                pred_val = float(st.session_state['model'].predict(X_single_sc)[0])
                use_target = st.session_state['target_variable']
                
                # ▼ 표시명 매핑 (RUL / Capacity / SoH로 보이게)
                def to_display_name(name: str) -> str:
                    m = {
                        "rul": "RUL",
                        "corr_capacity2": "Capacity",
                        "capacity": "Capacity",
                        "soh2": "SoH",
                        "soh": "SoH",
                        "soh%": "SoH",
                    }
                    return m.get(str(name).lower(), str(name))

                display_name = to_display_name(use_target)

                st.success("예측이 완료되었습니다.")

                # 단위 설정
                if str(use_target).lower() == "soh2":
                    unit = "%"
                    pct = clamp(pred_val, 0, 100)
                elif str(use_target).lower() == "corr_capacity2":
                    unit = "Ah"
                    stats = st.session_state.get('target_stats', None)
                    baseline = (stats.get('p95') or stats.get('max') or pred_val) if stats else pred_val
                    baseline = max(1e-6, float(baseline))
                    pct = max(0.0, min(100.0, (pred_val / baseline) * 100.0))
                else:  # RUL 등
                    unit = "cycles"
                    stats = st.session_state.get('target_stats', None)
                    baseline = (stats.get('p95') or stats.get('max') or pred_val) if stats else pred_val
                    baseline = max(1e-6, float(baseline))
                    pct = max(0.0, min(100.0, (pred_val / baseline) * 100.0))

                # 화면 중앙용 컨테이너 CSS
                st.markdown(
                    """
                    <style>
                    .center-row {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        gap: 40px;  /* 애니메이션과 숫자 사이 간격 */
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # 화면 중앙에 방전 애니메이션 + 숫자 표시
                with st.container():
                    st.markdown('<div class="center-row">', unsafe_allow_html=True)

                    components.html(
                        render_battery(
                            pct,
                            title=f"남은 수명 ({display_name})",
                            subtitle="",  # subtitle 제거
                            height=260, min_width=700, max_width=1200, mode="drain",
                            right_value_text=f"""
                                            <div style='
                                                position:absolute;
                                                top:40%;
                                                left:40%;
                                                transform:translate(-50%, -50%);
                                                font-weight:bold;
                                                font-size:48px;      
                                                color:black;                                       
                                                -webkit-text-stroke: 2px black;
                                                text-stroke: 2px black;
                                            '>
                                                {pred_val:.1f} {unit}
                                            </div>
                                            """,
                            right_caption=""  # caption 제거
                        ),
                        height=460, width=1100, scrolling=False
                    )


# --------------------------------------------------------------------------------
# Pages Routing (1: Realtime, 2: Main, 3: Prediction)
# --------------------------------------------------------------------------------

# 1) Real-time Dashboard (기본 1페이지)
if st.session_state.page == 'realtime':
    st.title("실시간 배터리 모니터링")
    st.markdown("---")

    # 업로드 섹션
    st.subheader("데이터 업로드")
    col1, col2 = st.columns(2)
    with col1:
        metadata_file = st.file_uploader("메타데이터 파일 업로드 (CSV)", type="csv",
                                         key="realtime_metadata", help="metadata.csv 파일을 업로드하세요")
    with col2:
        if metadata_file is not None:
            experiment_files = st.file_uploader("실험 데이터 업로드 (CSV)", type="csv",
                                                key="realtime_experiment",
                                                help="실험 파일들을 업로드하세요 (00001.csv ~ 07565.csv)",
                                                accept_multiple_files=True)
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("먼저 메타데이터 파일을 업로드해주세요")
            experiment_files = []
    st.markdown("---")

    # 세션 상태 기본값
    if 'battery_data' not in st.session_state:
        st.session_state.battery_data = {}
    if 'is_monitoring' not in st.session_state:
        st.session_state.is_monitoring = True
    st.session_state.setdefault('realtime_v_min', 2.5)
    st.session_state.setdefault('realtime_v_max', 4.2)
    st.session_state.setdefault('realtime_t_max', 60)
    st.session_state.setdefault('realtime_speed', 1.0)
    st.session_state.setdefault('realtime_points', 10)

    # 메타데이터 처리
    metadata_df = None
    if metadata_file is not None:
        try:
            metadata_df = pd.read_csv(metadata_file)
            st.success("메타데이터 파일이 성공적으로 업로드되었습니다!")
            if 'start_time' in metadata_df.columns:
                start_time_str = metadata_df['start_time'].astype(str).str.slice(1, -1)
                start_time_split = start_time_str.str.split()
                start_time_fixed = start_time_split.apply(process_row)
                start_time_datetime = start_time_fixed.apply(datetime_change)
                metadata_df['start_time'] = start_time_datetime.dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            st.error(f"메타데이터 처리 오류: {str(e)}")
            metadata_df = None

    # 실험 파일 처리
    if 'experiment_files' not in locals():
        experiment_files = []
    if experiment_files and metadata_df is not None:
        try:
            for experiment_file in experiment_files:
                experiment_filename = experiment_file.name
                experiment_df = pd.read_csv(experiment_file)

                matched_rows = None
                if 'filename' in metadata_df.columns:
                    matched_rows = metadata_df[metadata_df['filename'] == experiment_filename]

                if matched_rows is not None and not matched_rows.empty:
                    battery_id = matched_rows['battery_id'].iloc[0]
                    test_id = matched_rows['test_id'].iloc[0]
                    uid = matched_rows['uid'].iloc[0]
                    data_type = matched_rows['type'].iloc[0] if 'type' in matched_rows.columns else 'unknown'
                    start_time = matched_rows['start_time'].iloc[0]

                    if str(data_type).strip().lower() in ['charge', 'discharge']:
                        required_cols = ['Time', 'Voltage_measured', 'Current_measured', 'Temperature_measured']
                        missing_cols = [col for col in required_cols if col not in experiment_df.columns]
                        if not missing_cols:
                            experiment_df = experiment_df.sort_values(by='Time').reset_index(drop=True)
                            if battery_id not in st.session_state.battery_data:
                                st.session_state.battery_data[battery_id] = {
                                    'experiment_df': experiment_df,
                                    'current_data_index': 0,
                                    'times': [],
                                    'voltages': [],
                                    'currents': [],
                                    'temperatures': [],
                                    'all_data_rows': [],
                                    'capacity': 0.0,
                                    'metadata': {
                                        'test_id': test_id,
                                        'uid': uid,
                                        'data_type': data_type,
                                        'start_time': start_time,
                                        'filename': experiment_filename
                                    }
                                }
                                st.success(f"배터리 {battery_id}를 모니터링에 추가했습니다")
                        else:
                            st.warning(f"{experiment_filename}에 누락된 컬럼: {missing_cols}")
                    else:
                        st.info(f"{experiment_filename} 건너뜀 (데이터 유형: {data_type}) - charge/discharge만 지원됩니다")
                else:
                    st.error(f"'{experiment_filename}'에 대한 일치하는 메타데이터를 찾을 수 없습니다")
        except Exception as e:
            st.error(f"실험 데이터 처리 오류: {str(e)}")

    # 화면 출력
    if st.session_state.battery_data:
        battery_ids = list(st.session_state.battery_data.keys())

        # 상단 실험 정보
        if battery_ids:
            initial_battery = st.session_state.get('selected_battery', battery_ids[0])
            if initial_battery in st.session_state.battery_data:
                metadata = st.session_state.battery_data[initial_battery]['metadata']
                st.subheader("실험 정보")
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1: st.markdown(f"<div style='font-size:28px'><strong>배터리 ID</strong><br>{initial_battery}</div>", unsafe_allow_html=True)
                with c2: st.markdown(f"<div style='font-size:28px'><strong>테스트 ID</strong><br>{metadata['test_id']}</div>", unsafe_allow_html=True)
                with c3: st.markdown(f"<div style='font-size:28px'><strong>UID</strong><br>{metadata['uid']}</div>", unsafe_allow_html=True)
                with c4: st.markdown(f"<div style='font-size:28px'><strong>데이터 유형</strong><br>{metadata['data_type']}</div>", unsafe_allow_html=True)
                with c5: st.markdown(f"<div style='font-size:28px'><strong>시작 시간</strong><br>{metadata['start_time']}</div>", unsafe_allow_html=True)
                st.markdown("---")

        # 데이터 한 스텝씩 진행
        updated_batteries = []
        for battery_id, bdata in st.session_state.battery_data.items():
            if st.session_state.is_monitoring and bdata['current_data_index'] < len(bdata['experiment_df']):
                row = bdata['experiment_df'].iloc[bdata['current_data_index']]
                bdata['times'].append(row['Time'])
                bdata['voltages'].append(row['Voltage_measured'])
                bdata['currents'].append(row['Current_measured'])
                bdata['temperatures'].append(row['Temperature_measured'])
                bdata['all_data_rows'].append(row.to_dict())

                if len(bdata['times']) > 1:
                    times_sec = np.array(bdata['times'])
                    if np.issubdtype(times_sec.dtype, np.datetime64):
                        times_sec = (times_sec - times_sec[0]) / np.timedelta64(1, 's')
                    currents_a = np.array(bdata['currents'])
                    capacity_ah = np.trapz(currents_a, x=times_sec) / 3600.0
                    bdata['capacity'] = abs(capacity_ah)

                bdata['current_data_index'] += 1
                updated_batteries.append(battery_id)

        # 선택 배터리
        if 'selected_battery' not in st.session_state and st.session_state.battery_data:
            st.session_state.selected_battery = battery_ids[0]
        selected_battery = st.session_state.get('selected_battery', battery_ids[0] if battery_ids else None)

        # 차트 표시
        if selected_battery and selected_battery in st.session_state.battery_data:
            bdata = st.session_state.battery_data[selected_battery]
            times = bdata['times']; voltages = bdata['voltages']; currents = bdata['currents']; temperatures = bdata['temperatures']

            if len(times) > 0:
                st.subheader("실시간 차트")

                current_voltage = voltages[-1]; prev_voltage = voltages[-2] if len(voltages) > 1 else current_voltage
                current_current = currents[-1]; prev_current = currents[-2] if len(currents) > 1 else currents[-1]
                current_temp = temperatures[-1]; prev_temp = temperatures[-2] if len(temperatures) > 1 else temperatures[-1]

                voltage_change = current_voltage - prev_voltage
                current_change = current_current - prev_current
                temp_change = current_temp - prev_temp

                voltage_min = st.session_state.realtime_v_min
                voltage_max = st.session_state.realtime_v_max
                temp_max = st.session_state.realtime_t_max
                max_points = st.session_state.get('realtime_points', 500)

                voltage_violation = current_voltage < voltage_min or current_voltage > voltage_max
                temp_violation = current_temp > temp_max

                if len(times) > max_points:
                    times = times[-max_points:]; voltages = voltages[-max_points:]; currents = currents[-max_points:]; temperatures = temperatures[-max_points:]
                if len(bdata['all_data_rows']) > 100:
                    bdata['all_data_rows'] = bdata['all_data_rows'][-100:]

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    arrow = "🟢" if voltage_change > 0 else ("🔴" if voltage_change < 0 else "🟢")
                    change_color = "color: green;" if voltage_change > 0 else ("color: red;" if voltage_change < 0 else "")
                    card_color = "color: red;" if voltage_violation else ""
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; border: {'2px solid red' if voltage_violation else '1px solid #ddd'};">
                        <h4 style="margin: 0; font-size: 32px; {card_color}">전압</h4>
                        <h2 style="margin: 5px 0; font-size: 32px; {card_color}">{current_voltage:.3f} V</h2>
                        <p style="margin: 0; font-size: 24px; {change_color}">{arrow} {voltage_change:+.3f} V</p>
                    </div>
                    """, unsafe_allow_html=True)

                with metric_col2:
                    arrow = "🟢" if current_change > 0 else ("🔴" if current_change < 0 else "🟢")
                    change_color = "color: green;" if current_change > 0 else ("color: red;" if current_change < 0 else "")
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #ddd;">
                        <h4 style="margin: 0; font-size: 32px;">전류</h4>
                        <h2 style="margin: 5px 0; font-size: 32px;">{current_current:.3f} A</h2>
                        <p style="margin: 0; font-size: 24px; {change_color}">{arrow} {current_change:+.3f} A</p>
                    </div>
                    """, unsafe_allow_html=True)

                with metric_col3:
                    arrow = "🟢" if temp_change > 0 else ("🔴" if temp_change < 0 else "🟢")
                    change_color = "color: green;" if temp_change > 0 else ("color: red;" if temp_change < 0 else "")
                    card_color = "color: red;" if temp_violation else ""
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; border: {'2px solid red' if temp_violation else '1px solid #ddd'};">
                        <h4 style="margin: 0; font-size: 32px; {card_color}">온도</h4>
                        <h2 style="margin: 5px 0; font-size: 32px; {card_color}">{current_temp:.1f} °C</h2>
                        <p style="margin: 0; font-size: 24px; {change_color}">{arrow} {temp_change:+.1f} °C</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                chart_col1, chart_col2, chart_col3 = st.columns(3)

                with chart_col1:
                    fig_voltage = go.Figure()
                    voltage_colors = ['red' if (v < voltage_min or v > voltage_max) else 'blue' for v in voltages]
                    fig_voltage.add_trace(go.Scatter(x=times, y=voltages, mode='lines+markers',
                                                     name='전압', line=dict(color='#1f77b4', width=2),
                                                     marker=dict(size=8, color=voltage_colors)))
                    if len(voltages) >= 2:
                        mean_v = np.mean(voltages); std_v = np.std(voltages, ddof=1)
                        ucl_v, lcl_v = mean_v + 3*std_v, mean_v - 3*std_v
                        fig_voltage.add_trace(go.Scatter(x=times, y=[mean_v]*len(times), mode='lines', name='CL', line=dict(color='black', dash='dash')))
                        fig_voltage.add_trace(go.Scatter(x=times, y=[ucl_v]*len(times), mode='lines', name='UCL', line=dict(color='red', dash='dot')))
                        fig_voltage.add_trace(go.Scatter(x=times, y=[lcl_v]*len(times), mode='lines', name='LCL', line=dict(color='red', dash='dot')))
                    fig_voltage.update_layout(xaxis=dict(title=dict(text='시간 (s)', font=dict(size=18, color='black')), 
                                                        tickfont=dict(size=18, color='black')),
                                              yaxis=dict(title=dict(text='전압 (V)', font=dict(size=18, color='black')),
                                                        tickfont=dict(size=18, color='black')),
                                              height=300, showlegend=False, margin=dict(l=50, r=40, t=50, b=50),
                                              plot_bgcolor='white', paper_bgcolor='white',
                                              font=dict(size=14, color='black'))
                    st.plotly_chart(fig_voltage, use_container_width=True)

                with chart_col2:
                    fig_current = go.Figure()
                    fig_current.add_trace(go.Scatter(x=times, y=currents, mode='lines+markers',
                                                     name='전류', line=dict(color='#2ca02c', width=2),
                                                     marker=dict(size=8)))
                    if len(currents) >= 2:
                        mean_c = np.mean(currents); std_c = np.std(currents, ddof=1)
                        ucl_c, lcl_c = mean_c + 3*std_c, mean_c - 3*std_c
                        fig_current.add_trace(go.Scatter(x=times, y=[mean_c]*len(times), mode='lines', name='CL', line=dict(color='black', dash='dash')))
                        fig_current.add_trace(go.Scatter(x=times, y=[ucl_c]*len(times), mode='lines', name='UCL', line=dict(color='red', dash='dot')))
                        fig_current.add_trace(go.Scatter(x=times, y=[lcl_c]*len(times), mode='lines', name='LCL', line=dict(color='red', dash='dot')))
                    fig_current.update_layout(xaxis=dict(title=dict(text='시간 (s)', font=dict(size=18, color='black')), 
                                                        tickfont=dict(size=18, color='black')),
                                              yaxis=dict(title=dict(text='전류 (A)', font=dict(size=18, color='black')),
                                                        tickfont=dict(size=18, color='black')),
                                              height=300, showlegend=False, margin=dict(l=50, r=40, t=50, b=50),
                                              plot_bgcolor='white', paper_bgcolor='white',
                                              font=dict(size=14, color='black'))
                    st.plotly_chart(fig_current, use_container_width=True)

                with chart_col3:
                    fig_temperature = go.Figure()
                    temp_colors = ['red' if (t > temp_max) else 'orange' for t in temperatures]
                    fig_temperature.add_trace(go.Scatter(x=times, y=temperatures, mode='lines+markers',
                                                         name='온도', line=dict(color='#ff7f0e', width=2),
                                                         marker=dict(size=8, color=temp_colors)))
                    fig_temperature.update_layout(xaxis=dict(title=dict(text='시간 (s)', font=dict(size=18, color='black')), 
                                                            tickfont=dict(size=18, color='black')),
                                                  yaxis=dict(title=dict(text='온도 (°C)', font=dict(size=18, color='black')),
                                                            tickfont=dict(size=18, color='black')),
                                                  height=300, showlegend=False, margin=dict(l=50, r=40, t=50, b=50),
                                                  plot_bgcolor='white', paper_bgcolor='white',
                                                  font=dict(size=14, color='black'))
                    st.plotly_chart(fig_temperature, use_container_width=True)

                st.markdown("---")

        # 하단: 오른쪽 로그 / 왼쪽 컨트롤
        bottom_left, bottom_right = st.columns([3, 3])

        with bottom_right:
            st.subheader("실험 데이터 로그")
            if selected_battery and selected_battery in st.session_state.battery_data:
                bdata = st.session_state.battery_data[selected_battery]
                if len(bdata['all_data_rows']) > 0:
                    recent_rows = list(reversed(bdata['all_data_rows'][-20:]))
                    voltage_min = st.session_state.realtime_v_min
                    voltage_max = st.session_state.realtime_v_max
                    temp_max = st.session_state.realtime_t_max
                    table_html = f"""
                    <div style="height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 5px; padding: 10px; background-color: #f9f9f9;" id="data-log-table">
                        <div style="font-weight: bold; margin-bottom: 10px; font-size: 16px;">배터리 {selected_battery} - 최신 데이터</div>
                        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                            <thead style="position: sticky; top: 0; background-color: #f0f2f6; z-index: 1;">
                                <tr>
                                    <th style="border: 1px solid #ddd; padding: 10px; text-align: left; font-size: 14px;">시간 (s)</th>
                                    <th style="border: 1px solid #ddd; padding: 10px; text-align: left; font-size: 14px;">전압 (V)</th>
                                    <th style="border: 1px solid #ddd; padding: 10px; text-align: left; font-size: 14px;">전류 (A)</th>
                                    <th style="border: 1px solid #ddd; padding: 10px; text-align: left; font-size: 14px;">온도 (°C)</th>
                                </tr>
                            </thead>
                            <tbody>
                    """
                    for row_dict in recent_rows:
                        v = row_dict['Voltage_measured']; t = row_dict['Temperature_measured']
                        v_violation = v < voltage_min or v > voltage_max
                        t_violation = t > temp_max
                        row_bg = "#ffebee" if (v_violation or t_violation) else "white"
                        v_color = "color: red; font-weight: bold;" if v_violation else ""
                        t_color = "color: red; font-weight: bold;" if t_violation else ""
                        table_html += f"""
                            <tr style="background-color: {row_bg};">
                                <td style="border: 1px solid #ddd; padding: 8px; font-size: 14px;">{row_dict['Time']:.3f}</td>
                                <td style="border: 1px solid #ddd; padding: 8px; font-size: 14px; {v_color}">{v:.3f}</td>
                                <td style="border: 1px solid #ddd; padding: 8px; font-size: 14px;">{row_dict['Current_measured']:.3f}</td>
                                <td style="border: 1px solid #ddd; padding: 8px; font-size: 14px; {t_color}">{t:.1f}</td>
                            </tr>
                        """
                    table_html += """
                            </tbody>
                        </table>
                    </div>
                    """
                    try:
                        st.html(table_html)
                    except Exception:
                        import streamlit.components.v1 as components
                        components.html(table_html, height=420, scrolling=True)

        with bottom_left:
            sel_col, cap_col = st.columns([1, 1])
            with sel_col:
                st.markdown("**<span style='font-size: 20px;'>배터리 선택</span>**", unsafe_allow_html=True)
                if battery_ids:
                    st.markdown("<div style='font-size: 16px; margin-bottom: 5px;'>표시할 배터리 선택</div>", unsafe_allow_html=True)
                    selected_battery = st.selectbox("", sorted(battery_ids),
                                                    key="selected_battery",
                                                    help="실시간 차트에 표시할 배터리 데이터를 선택하세요",
                                                    label_visibility="collapsed")
            with cap_col:
                st.markdown("**<span style='font-size: 18px;'>용량 예측</span>**", unsafe_allow_html=True)
                if selected_battery in st.session_state.battery_data:
                    bid = selected_battery
                    battery_capacity = st.session_state.battery_data[bid]['capacity']
                    battery_progress = (
                        st.session_state.battery_data[bid]['current_data_index'] /
                        len(st.session_state.battery_data[bid]['experiment_df']) * 100
                    )
                    status = "완료" if battery_progress >= 100 else "🔋"
                    st.markdown(
                        f"<span style='font-size:38px'>{status} <b>{bid}</b>: {battery_capacity:.3f}Ah ({battery_progress:.1f}%)</span>",
                        unsafe_allow_html=True
                    )

            with st.expander("🚨 임계값", expanded=False):
                th_col1, th_col2 = st.columns(2)
                with th_col1:
                    st.markdown("<div style='font-size: 16px; margin-bottom: 5px;'>최소 전압 (V)</div>", unsafe_allow_html=True)
                    st.number_input("", 0.0, 5.0, st.session_state.realtime_v_min, key="realtime_v_min", label_visibility="collapsed")
                    st.markdown("<div style='font-size: 16px; margin-bottom: 5px;'>최대 전압 (V)</div>", unsafe_allow_html=True)
                    st.number_input("", 0.0, 5.0, st.session_state.realtime_v_max, key="realtime_v_max", label_visibility="collapsed")
                with th_col2:
                    st.markdown("<div style='font-size: 16px; margin-bottom: 5px;'>최대 온도 (°C)</div>", unsafe_allow_html=True)
                    st.number_input("", 0, 100, st.session_state.realtime_t_max, key="realtime_t_max", label_visibility="collapsed")

            with st.expander("⚙️ 설정", expanded=True):
                st.markdown("<div style='font-size: 16px; margin-bottom: 5px;'>재생 속도</div>", unsafe_allow_html=True)
                st.slider("", 0.1, 10.0, st.session_state.realtime_speed, 0.1, key="realtime_speed", label_visibility="collapsed")
                st.markdown("<div style='font-size: 16px; margin-bottom: 5px;'>최대 표시 포인트</div>", unsafe_allow_html=True)
                st.number_input("", 10, 1000, st.session_state.realtime_points, key="realtime_points", label_visibility="collapsed")

        # 자동 리프레시
        if updated_batteries and st.session_state.is_monitoring:
            if selected_battery in updated_batteries:
                bdata = st.session_state.battery_data[selected_battery]
                if bdata['current_data_index'] < len(bdata['experiment_df']):
                    current_idx = bdata['current_data_index'] - 1
                    if current_idx >= 0 and bdata['current_data_index'] < len(bdata['experiment_df']):
                        current_time = bdata['experiment_df'].iloc[current_idx]['Time']
                        next_time = bdata['experiment_df'].iloc[bdata['current_data_index']]['Time']
                        delay = next_time - current_time
                        if delay <= 0:
                            delay = 0.1
                        actual_delay = min(delay / st.session_state.realtime_speed, 5.0)
                        time.sleep(actual_delay)
            _safe_rerun()

        # 완료 체크
        all_complete = all(
            bdata['current_data_index'] >= len(bdata['experiment_df'])
            for bdata in st.session_state.battery_data.values()
        )
        if all_complete and st.session_state.is_monitoring:
            st.success("모든 배터리 모니터링이 완료되었습니다!")
    else:
        st.markdown("""
        ### <span style='font-size: 22px;'>사용 안내</span>
        <div style='font-size: 16px;'>
        1. <strong>메타데이터 파일</strong> (metadata.csv)을 먼저 업로드하세요<br>  
        2. <strong>여러 실험 데이터 파일</strong> (00001.csv ~ 07565.csv)을 업로드하세요<br>  
        3. 시스템이 자동으로 파일을 매칭하고 모든 배터리에 대해 실시간 모니터링을 시작합니다<br>  
        4. <strong>드롭다운을 사용하여</strong> 차트에 표시할 배터리를 선택하세요<br>  
        5. 설정에서 재생 속도와 임계값을 조정하세요<br>  
        6. 선택과 관계없이 모든 배터리가 백그라운드에서 계속 업데이트됩니다
        </div>

        ### <span style='font-size: 22px;'>데이터 형식 요구사항</span>
        <div style='font-size: 16px;'>
        - <strong>메타데이터</strong>: <code>filename</code>, <code>battery_id</code>, <code>test_id</code>, <code>uid</code>, <code>start_time</code> 컬럼 필수<br>
        - <strong>실험 데이터</strong>: <code>Time</code>, <code>Voltage_measured</code>, <code>Current_measured</code>, <code>Temperature_measured</code> 컬럼 필수
        </div>
        """, unsafe_allow_html=True)

# 2) Main Dashboard (2페이지)
elif st.session_state.page == 'main':
    # 1. Main Dashboard 제목과 파일 업로드를 같은 행에 배치
    col_title, col_spacer, col_upload = st.columns([2, 2, 1.5])
    with col_title:
        st.markdown("# 메인 대시보드")
    with col_upload:
        cycle_file = st.file_uploader("CSV 업로드", type=['csv'], key="cycle_upload")
        if cycle_file:
            st.session_state.cycle_data = pd.read_csv(cycle_file)
            st.success("파일이 업로드되었습니다.")

    if st.session_state.cycle_data is not None:
        df = st.session_state.cycle_data

        # 왼쪽(그룹/조건) - 오른쪽(SoH)
        col1, col2 = st.columns([2, 4])
        with col1:
            col1_left, col1_right = st.columns([1.5, 1])
            with col1_left:
                st.markdown("<div style='font-size:26px; font-weight:700;'>그룹 필터링</div>", unsafe_allow_html=True)
                group_keys = list(BATTERY_GROUPS.keys())
                selected_group = st.selectbox("배터리 그룹 선택", group_keys, index=0)
                battery_ids = BATTERY_GROUPS[selected_group]
                df_filtered = df[df['battery_id'].isin(battery_ids)]

            st.markdown("---")
            exp_col1, divider_col, exp_col2 = st.columns([1.2, 0.1, 1])
            with exp_col1:
                st.markdown("<div style='font-size:26px; font-weight:700;'>실험 조건</div>", unsafe_allow_html=True)
                if not df_filtered.empty:
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        avg_temp = df_filtered['ambient_temperature'].mean() if 'ambient_temperature' in df_filtered else 0
                        st.metric("온도", f"{avg_temp:.1f}°C", label_visibility="visible")
                    with metric_col2:
                        if 'current_load' in df_filtered:
                            current_vals = df_filtered['current_load'].unique()
                            st.metric("전류", f"{current_vals[0] if len(current_vals) > 0 else 'N/A'}A", label_visibility="visible")
                    with metric_col3:
                        if 'cutoff_voltage' in df_filtered:
                            cutoff_vals = df_filtered['cutoff_voltage'].unique()
                            st.metric("전압", f"{cutoff_vals[0] if len(cutoff_vals) > 0 else 'N/A'}V", label_visibility="visible")
            with divider_col:
                st.markdown("<div style='height: 100px; width: 1px; background-color: #e5e7eb; margin: 20px auto;'></div>", unsafe_allow_html=True)
            with exp_col2:
                st.markdown("<div style='font-size:26px; font-weight:700;'>평균 발열량</div>", unsafe_allow_html=True)
                if not df_filtered.empty and 'total_heat_joules' in df_filtered:
                    avg_heat = df_filtered.groupby('battery_id')['total_heat_joules'].mean().mean()
                    st.metric("평균 열량", f"{avg_heat:.2f} J")
                else:
                    st.info("발열 데이터가 없습니다.")

        with col2:
            st.markdown("""
            <div style="display:flex; align-items:center; gap:12px;">
            <h2 style="font-size:26px; font-weight:700; margin:0;">배터리별 SoH 변화</h2>
            <span style="font-size:18px; color:#111827; display:flex; align-items:center; gap:6px;">
                <span style="display:inline-block; width:12px; height:12px; background:#10b981; border-radius:2px;"></span>
                : 최종 배터리 SoH
            </span>
            </div>
            """, unsafe_allow_html=True)

            if not df_filtered.empty and 'battery_id' in df_filtered and 'SoH' in df_filtered:
                soh_data, text_info = [], []
                sorted_battery_ids = sorted(df_filtered['battery_id'].unique())
                for battery_id in sorted_battery_ids:
                    battery_data = df_filtered[df_filtered['battery_id'] == battery_id]
                    if not battery_data.empty:
                        initial_soh = min(battery_data['SoH'].iloc[0], 100)
                        current_soh = battery_data['SoH'].iloc[-1]
                        remaining = current_soh
                        lost = initial_soh - current_soh
                        never_available = 100 - initial_soh
                        soh_data.append({'Battery ID': battery_id, 'Remaining': remaining, 'Lost': lost, 'Never_Available': never_available})
                        text_info.append({'Battery ID': battery_id, 'Initial': initial_soh, 'Current': current_soh, 'Change': initial_soh-current_soh})
                if soh_data:
                    soh_df = pd.DataFrame(soh_data); text_df = pd.DataFrame(text_info)
                    for i, (_, row) in enumerate(soh_df.iterrows()):
                        text_row = text_df.iloc[i]; battery_id = row['Battery ID']
                        c1, c2 = st.columns([3, 1.5])
                        with c1:
                            fig = go.Figure()
                            fig.add_trace(go.Bar(y=[battery_id], x=[row['Remaining']], orientation='h', marker_color='#10b981', showlegend=False))
                            fig.add_trace(go.Bar(y=[battery_id], x=[row['Lost']], orientation='h', marker_color='#FCA5A5', showlegend=False))
                            fig.add_trace(go.Bar(y=[battery_id], x=[row['Never_Available']], orientation='h', marker_color='#9CA3AF', showlegend=False))
                            show_x_axis = (i == len(soh_df) - 1)
                            fig.update_layout(barmode='stack', height=65, showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
                                              plot_bgcolor='white', paper_bgcolor='white',
                                              xaxis=dict(showticklabels=show_x_axis, showline=show_x_axis, zeroline=False),
                                              yaxis=dict(tickfont=dict(size=14, family="Arial Black")), bargap=0)
                            st.plotly_chart(fig, use_container_width=True)
                        with c2:
                            st.markdown(f"""
                            <div style='text-align: center; font-size: 25px; font-weight: bold; margin-top:15px;'>
                                <span style='color: #e5e7eb !important;'>{text_row['Initial']:.1f}%</span> →
                                <span style='color: #10b981;'>{text_row['Current']:.1f}%</span>
                                (<span style='color: #dc2626;'>-{text_row['Change']:.1f}%</span>)
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.warning("SoH 데이터가 없습니다.")
                
        st.markdown("---")
        
        # 메인 차트 2개
        c1, mid, c2 = st.columns([1, 0.03, 1])
        with c1:
            st.markdown("""
            <style>
            h2, h3 { margin-bottom: 6px; }  /* subheader가 h3로 렌더링되면 h3에 적용 */
            </style>
            """, unsafe_allow_html=True)
            st.subheader("용량")
            if not df_filtered.empty and 'cycle' in df_filtered and 'Capacity' in df_filtered:
                fig = go.Figure()
                for bid in df_filtered['battery_id'].unique()[:5]:
                    bd = df_filtered[df_filtered['battery_id'] == bid]
                    fig.add_trace(go.Scatter(x=bd['cycle'], y=bd['Capacity'], mode='lines', name=bid))
                fig.update_layout(xaxis_title='사이클 수', yaxis_title='용량 (Ah)', height=400,
                                plot_bgcolor='white', paper_bgcolor='white',
                                xaxis=dict(tickfont=dict(size=18, color='black'), title_font=dict(size=18, color='black')),
                                yaxis=dict(tickfont=dict(size=18, color='black'), title_font=dict(size=18, color='black'))
                                )
                st.plotly_chart(fig, use_container_width=True)
                
        with mid:
            LINE_TOP = 40          # px, 원하는 만큼 아래로 내림
            LINE_H   = 400         # 차트 height와 비슷하게
            LINE_COL = "#e5e7eb"   # 회색 
            st.markdown(
                f"<div style='height:{LINE_H}px; width:1px; "
                f"background-color:{LINE_COL}; margin:{LINE_TOP}px auto 0;'></div>",
                unsafe_allow_html=True
            )

        with c2:
            st.markdown("""
            <style>
            h2, h3 { margin-bottom: 6px; }  /* subheader가 h3로 렌더링되면 h3에 적용 */
            </style>
            """, unsafe_allow_html=True)
            st.markdown('<div class="v-divider-left">', unsafe_allow_html=True)
            st.subheader("전체 저항")
            if not df_filtered.empty and 'cycle' in df_filtered and 'total_R' in df_filtered:
                fig = go.Figure()
                for bid in df_filtered['battery_id'].unique()[:5]:
                    bd = df_filtered[df_filtered['battery_id'] == bid]
                    fig.add_trace(go.Scatter(x=bd['cycle'], y=bd['total_R'], mode='lines', name=bid))
                fig.update_layout(xaxis_title='사이클 수', yaxis_title='전체 저항 (Ω)', height=400,
                                plot_bgcolor='white', paper_bgcolor='white',
                                xaxis=dict(tickfont=dict(size=18, color='black'), title_font=dict(size=18, color='black')),
                                yaxis=dict(tickfont=dict(size=18, color='black'), title_font=dict(size=18, color='black')))
                st.plotly_chart(fig, use_container_width=True)
                
        st.markdown("---")

        # 그룹 요약
        st.markdown("### 그룹 성능 요약")
        selected_groups = {k: v for k, v in BATTERY_GROUPS.items() if not any(x in k for x in ['Group 4', 'Group 5', 'Group 6', 'Group 8'])}
        col_header1, col_header2, col_header3, col_header4 = st.columns([2.5, 1.5, 0.5, 0.5])
        with col_header2:
            st.markdown("<div style='text-align: center; font-size: 21px; font-weight: bold;'>평균 SoH 변화</div>", unsafe_allow_html=True)
        with col_header3:
            st.markdown("<div style='text-align: center; font-size: 21px; font-weight: bold;'>사이클당 SoH 변화</div>", unsafe_allow_html=True)
        for i, (group_name, battery_ids) in enumerate(selected_groups.items()):
            group_data = df[df['battery_id'].isin(battery_ids)] if 'battery_id' in df.columns else pd.DataFrame()
            if not group_data.empty and 'SoH' in group_data and 'cycle' in group_data:
                initial_soh_avg = 0; current_soh_avg = 0; valid_batteries = 0
                for bid in battery_ids:
                    bd = group_data[group_data['battery_id'] == bid]
                    if not bd.empty:
                        initial_soh = min(bd['SoH'].iloc[0], 100)
                        current_soh = bd['SoH'].iloc[-1]
                        initial_soh_avg += initial_soh; current_soh_avg += current_soh; valid_batteries += 1
                if valid_batteries > 0:
                    initial_soh_avg /= valid_batteries; current_soh_avg /= valid_batteries
                    soh_change_avg = initial_soh_avg - current_soh_avg
                    remaining_capacity = current_soh_avg
                    lost_capacity = initial_soh_avg - current_soh_avg
                    never_available = 100 - initial_soh_avg
                    avg_soh_change = 0
                    for bid in battery_ids:
                        bd = group_data[group_data['battery_id'] == bid]
                        if not bd.empty and len(bd) > 1:
                            cycles = bd['cycle'].max() - bd['cycle'].min()
                            soh_change = bd['SoH'].iloc[0] - bd['SoH'].iloc[-1]
                            if cycles > 0: avg_soh_change += soh_change / cycles
                    avg_soh_change = avg_soh_change / len(battery_ids) if battery_ids else 0

                    c1, c2, c3, _ = st.columns([2.5, 1.5, 0.5, 0.5])
                    with c1:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(y=[group_name.split('(')[0]], x=[remaining_capacity], orientation='h', marker_color='#10b981', showlegend=False))
                        fig.add_trace(go.Bar(y=[group_name.split('(')[0]], x=[lost_capacity], orientation='h', marker_color='#FCA5A5', showlegend=False))
                        fig.add_trace(go.Bar(y=[group_name.split('(')[0]], x=[never_available], orientation='h', marker_color='#9CA3AF', showlegend=False))
                        show_x_axis = (i == len(selected_groups) - 1)
                        fig.update_layout(barmode='stack', height=65, showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
                                          plot_bgcolor='white', paper_bgcolor='white',
                                          xaxis=dict(showticklabels=show_x_axis, showline=show_x_axis, zeroline=False),
                                          yaxis=dict(tickfont=dict(size=14, family="Arial Black")), bargap=0)
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        st.markdown(f"""
                        <div style='text-align: center; font-size: 30px; font-weight: bold; margin-top: 15px;'>
                            <span style='color: #e5e7eb !important;'>{initial_soh_avg:.1f}%</span> →
                            <span style='color: #10b981;'>{current_soh_avg:.1f}%</span>
                            (<span style='color: #dc2626;'>-{soh_change_avg:.1f}%</span>)
                        </div>
                        """, unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"<div style='text-align: center; font-size: 30px; font-weight: bold; color: #000080; margin-top: 15px;'>{avg_soh_change:.4f}%</div>", unsafe_allow_html=True)
            else:
                c1, c2, c3 = st.columns([2.5, 1.2, 1])
                with c1:
                    st.info(f"{group_name.split('(')[0]} - 데이터가 없습니다.")
                with c2:
                    st.markdown("<div style='text-align: center; margin-top: 35px;'>N/A</div>", unsafe_allow_html=True)
                with c3:
                    st.markdown("<div style='text-align: center; margin-top: 35px;'>N/A</div>", unsafe_allow_html=True)
    else:
        st.info("대시보드를 보려면 사이클 데이터 CSV 파일을 업로드하세요.")

# 3) Predictive Modeling (3페이지)
elif st.session_state.page == 'prediction':
    run_predictive_modeling()

# --------------------------------------------------------------------------------
# Footer (공통)
# --------------------------------------------------------------------------------
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: #6b7280; font-size: 0.875rem;'>
        배터리 품질 대시보드 v1.0 | 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
""", unsafe_allow_html=True)
