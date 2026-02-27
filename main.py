import streamlit as st
import pandas as pd
from io import BytesIO
from src.classifier import classify_buzz_revelent, classify_buzz_category

PROJECT_LIST = ["ShopeeFood", "Shopee", "SPX Express", "Giao Hàng Nhanh"]

if "disabled" not in st.session_state:
    st.session_state["disabled"] = False

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

if "df_input" not in st.session_state:
    st.session_state["df_input"] = None

if "df_result_task_1" not in st.session_state:
    st.session_state["df_result_task_1"] = None

if "df_result_task_2" not in st.session_state:
    st.session_state["df_result_task_2"] = None


def reset_app():
    st.session_state["df_input"] = None
    st.session_state["df_result_task_1"] = None
    st.session_state["df_result_task_2"] = None
    st.session_state["disabled"] = False
    st.session_state["uploader_key"] += 1
    st.rerun()


def download_excel(df: pd.DataFrame, filename: str):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Result")
    output.seek(0)

    st.download_button(
        "⬇️ Tải file kết quả",
        output,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def app():
    
    st.set_page_config(
        page_title="Phân loại dữ liệu",
        layout="centered"
    )

    st.markdown("""
    <style>
    .big-task-title {
        font-size: 20px;
        font-weight: 800;
        margin: 10px 0 15px 0;
    }

    div[data-baseweb="tab-list"] {
        display: flex;
        margin-bottom: 25px;
        
    }

    button[data-baseweb="tab"] {
        flex: 1;
        text-align: center;
        font-weight: 800;
        font-size: 22px !important;
        height: 50px;
        border-radius: 12px 12px 0 0;
    }

    button[data-baseweb="tab"]:hover {
        background-color: #dbeafe;
    }

    </style>
    """, unsafe_allow_html=True)

    st.title("📊 Phân loại dữ liệu Excel")

    st.divider()
    if st.button("🆕 Tạo mới"):
        reset_app()
    st.divider()

    st.subheader("📤 Upload file Excel")

    uploaded_file = st.file_uploader(
        "Chọn file Excel",
        type=["xlsx"],
        disabled=st.session_state["disabled"],
        key=f"upload_shared_{st.session_state['uploader_key']}"
    )

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        cols = ["Title", "Content", "Description", "TopicId"]

        df[cols] = df[cols].fillna("").astype(str)

        st.session_state["df_input"] = df

    if st.session_state["df_input"] is not None:
        st.subheader("🔍 Dữ liệu gốc")
        st.dataframe(st.session_state["df_input"].head(100), width="stretch")

    st.markdown("---")

    tab_relevant, tab_seller_buyer = st.tabs([
        "Phân loại Relevant / Irrelevant",
        "Phân loại Seller / Buyer"
    ])
    
    with tab_relevant:
        st.markdown(
            '<div class="big-task-title">PHÂN LOẠI RELEVANT / IRRELEVANT</div>',
            unsafe_allow_html=True
        )

        if st.session_state["df_input"] is None:
            st.info("⬆️ Vui lòng upload file Excel trước")
        else:
            if st.button("⚙️ Xử lý", key="process_relevant"):
                with st.spinner("Đang phân loại Relevant / Irrelevant..."):
                   
                    st.session_state["df_result_task_1"] = classify_buzz_revelent(st.session_state["df_input"])
                    st.session_state["disabled"] = True

        if st.session_state["df_result_task_1"] is not None:
            st.success("✅ Xử lý xong!")
            st.dataframe(
                st.session_state["df_result_task_1"].head(100),
                width="stretch"
            )
            download_excel(
                st.session_state["df_result_task_1"],
                "ket_qua_relevant_irrelevant.xlsx"
            )
    
    with tab_seller_buyer:
        st.markdown(
            '<div class="big-task-title">PHÂN LOẠI SELLER / BUYER</div>',
            unsafe_allow_html=True
        )

        if st.session_state["df_input"] is None:
            st.info("⬆️ Vui lòng upload file Excel trước")
        else:
            selected_project = st.selectbox(
                "Chọn project",
                options=PROJECT_LIST
                )
            if selected_project:
                st.session_state["selected_project"] = selected_project
                if st.button("⚙️ Xử lý", key="process_seller_buyer"):
                    with st.spinner("Đang phân loại Seller / Buyer..."):
                        st.session_state["df_result_task_2"] = classify_buzz_category(
                            st.session_state["df_input"],
                            project_name=st.session_state["selected_project"]
                        )
                        st.session_state["disabled"] = True

        if st.session_state["df_result_task_2"] is not None:
            st.success("✅ Xử lý xong!")
            st.dataframe(
                st.session_state["df_result_task_2"].head(100),
                width="stretch"
            )
            download_excel(
                st.session_state["df_result_task_2"],
                "ket_qua_seller_buyer.xlsx"
            )


if __name__ == "__main__":
    app()
