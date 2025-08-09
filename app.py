# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import requests
from io import StringIO

# ✅ Cấu hình trang
st.set_page_config(page_title="📊 Dashboard Doanh Thu & Dự báo", page_icon="💼", layout="wide")

# Sidebar
st.sidebar.title("🔧 Tùy chọn")
selected_year = st.sidebar.selectbox("Chọn năm", [2022, 2023, 2024])
show_chart = st.sidebar.checkbox("Hiển thị biểu đồ", value=True)

# Tiêu đề chính
st.markdown("# 💼 **Dashboard Doanh Thu**")
st.markdown(f"📅 Dữ liệu năm: **{selected_year}**")

# Dữ liệu mẫu
data = {
    "Tháng": [f"Tháng {i}" for i in range(1, 13)],
    "Doanh thu": [12000 + i * 500 + (i % 3) * 1000 for i in range(1, 13)],
}
df_sample = pd.DataFrame(data)

# Tabs
tab1, tab2, tab3 = st.tabs(["📄 Dữ liệu", "📈 Biểu đồ", "📊 Tổng quan"])

with tab1:
    st.subheader("📄 Xem dữ liệu")
    st.dataframe(df_sample)

with tab2:
    if show_chart:
        st.subheader("📈 Biểu đồ doanh thu theo tháng")
        fig = px.line(df_sample, x="Tháng", y="Doanh thu", markers=True, title="Doanh thu theo tháng")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("🔕 Bạn đã tắt biểu đồ trong sidebar.")

with tab3:
    st.subheader("📊 Tổng quan")
    st.metric(label="📌 Tổng doanh thu", value=f"${df_sample['Doanh thu'].sum():,.0f}")
    st.metric(label="📈 Doanh thu cao nhất", value=f"${df_sample['Doanh thu'].max():,.0f}")
    st.metric(label="📉 Doanh thu thấp nhất", value=f"${df_sample['Doanh thu'].min():,.0f}")

# 🔗 Link CSV hợp lệ
RAW_CSV_URL = "https://raw.githubusercontent.com/trinh2115/nktrinh/main/Inventory%20Demand%20Forecast.csv"

# ✅ Tải dữ liệu có cache
@st.cache_data
def load_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        df = pd.read_csv(StringIO(response.text))
        return df
    else:
        return None

st.title("📊 Phân tích & Dự báo nhu cầu sản phẩm bằng LSTM")

with st.spinner("🔄 Đang tải dữ liệu từ GitHub..."):
    df = load_data(RAW_CSV_URL)
    if df is not None:
        st.success("✅ Dữ liệu đã được tải thành công!")

        # Xử lý dữ liệu
        df = df.dropna().drop_duplicates()
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

        st.subheader("📋 Thông tin dữ liệu")
        st.dataframe(df.head())

        # Tổng doanh thu mỗi ngày
        daily_sales = df.groupby('date')['total_sales'].sum().reset_index()

        st.subheader("📈 Xu hướng doanh thu theo ngày")
        fig1, ax1 = plt.subplots()
        ax1.plot(daily_sales['date'], daily_sales['total_sales'], marker='o', color='blue')
        ax1.set_title('Xu hướng doanh thu theo ngày')
        ax1.set_xlabel('Ngày')
        ax1.set_ylabel('Doanh thu (USD)')
        ax1.grid(True)
        st.pyplot(fig1)

        # Tỷ trọng doanh thu theo danh mục (giới hạn top 10)
        category_sales = df.groupby('categories')['total_sales'].sum().sort_values(ascending=False)
        top_categories = category_sales.head(10)
        st.subheader("📊 Tỷ trọng doanh thu theo danh mục")
        fig2, ax2 = plt.subplots()
        top_categories.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax2)
        ax2.set_ylabel('')
        st.pyplot(fig2)

        # Phân bố giá sản phẩm
        st.subheader("📦 Phân bố giá sản phẩm")
        fig3, ax3 = plt.subplots()
        df.boxplot(column='unit_price', ax=ax3)
        ax3.set_title('Phân bố giá sản phẩm')
        ax3.set_ylabel('Giá (USD)')
        st.pyplot(fig3)

        # Mối quan hệ giữa giá và số lượng bán
        st.subheader("💵 Giá vs Số lượng bán")
        fig4, ax4 = plt.subplots()
        df.plot(kind='scatter', x='unit_price', y='units_sold', alpha=0.6, color='green', ax=ax4)
        ax4.set_title('Giá vs Số lượng bán')
        st.pyplot(fig4)

        # Doanh thu theo nhà cung cấp
        supplier_sales = df.groupby('supplier')['total_sales'].sum().sort_values(ascending=False)
        st.subheader("🏭 Doanh thu theo nhà cung cấp")
        fig5, ax5 = plt.subplots()
        supplier_sales.plot(kind='bar', color='purple', ax=ax5)
        ax5.set_title('Doanh thu theo nhà cung cấp')
        ax5.set_ylabel('Doanh thu (USD)')
        ax5.set_xticklabels(supplier_sales.index, rotation=45)
        st.pyplot(fig5)

        # Dự báo bằng LSTM
        st.subheader("🔮 Dự báo nhu cầu sản phẩm")

        if 'forecasted_demand' in df.columns:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[['forecasted_demand']])

            def create_sequences(data, window_size):
                X, y = [], []
                for i in range(len(data) - window_size):
                    X.append(data[i:i+window_size])
                    y.append(data[i+window_size])
                return np.array(X), np.array(y)

            window_size = st.slider("🔢 Chọn kích thước cửa sổ thời gian", 5, 30, 10)
            X, y = create_sequences(scaled_data, window_size)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')

            with st.spinner("⏳ Đang huấn luyện mô hình..."):
                history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

            predicted = model.predict(X)
            predicted_original = scaler.inverse_transform(predicted)
            y_original = scaler.inverse_transform(y)

            mae = mean_absolute_error(y_original, predicted_original)
            mse = mean_squared_error(y_original, predicted_original)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_original, predicted_original)

            st.markdown(f"""
            **📌 Đánh giá mô hình:**
            - MAE: `{mae:.4f}`
            - MSE: `{mse:.4f}`
            - RMSE: `{rmse:.4f}`
            - R² Score: `{r2:.4f}`
            """)

            errors = y_original - predicted_original

            st.subheader("📉 Biểu đồ sai số dự đoán")
            fig6, ax6 = plt.subplots()
            ax6.plot(errors, color='red')
            ax6.set_title('Biểu đồ sai số dự đoán')
            st.pyplot(fig6)

            st.subheader("📊 Phân phối sai số")
            fig7, ax7 = plt.subplots()
            sns.histplot(errors.flatten(), kde=True, color='orange', ax=ax7)
            st.pyplot(fig7)
        else:
            st.warning("⚠️ Cột 'forecasted_demand' không tồn tại trong dữ liệu.")
    else:
        st.error("❌ Không thể tải dữ liệu. Kiểm tra lại link GitHub.")
