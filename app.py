# Full Sales Analysis Streamlit App (No Cleaning - Analysis Only)
# سيتم الآن بناء التطبيق كاملاً للقيام بجميع عمليات التحليل فقط.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly

st.set_page_config(page_title="Full Sales Analysis", layout="wide")
st.title("📊 نظام التحليل الكامل للبيانات (مبيعات)")

# ============================
# 1) تحميل الملف
# ============================
file = st.file_uploader("⬆️ قم برفع ملف المبيعات (CSV / Excel)")

if file:
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except:
        st.error("❌ خطأ في قراءة الملف")
        st.stop()

    st.subheader("🔍 معاينة البيانات")
    st.dataframe(df.head())

    # ============================
    # 2) اختيار أسماء الأعمدة يدويًا (يدعم عربي + إنجليزي)
    # ============================
    st.sidebar.title("⚙️ اختيار الأعمدة")
    col_product = st.sidebar.text_input("اسم عمود المنتج:")
    col_sales = st.sidebar.text_input("اسم عمود المبيعات:")
    col_qty = st.sidebar.text_input("اسم عمود الكمية:")
    col_price = st.sidebar.text_input("اسم عمود السعر:")
    col_profit = st.sidebar.text_input("اسم عمود الربح:")
    col_cost = st.sidebar.text_input("اسم عمود التكلفة الإجمالية:")
    col_date = st.sidebar.text_input("اسم عمود التاريخ (للتحليلات الزمنية):")

    if col_product and col_sales:

        # ============================
        # 3) جميع عمليات التحليل
        # ============================
        st.header("📈 التحليلات الأساسية")

        # أعلى منتج مبيعًا
        top_sales = df.groupby(col_product)[col_sales].sum().sort_values(ascending=False).head(10)

        fig1 = px.bar(top_sales, title="🏆 أعلى المنتجات مبيعًا (بناءً على المبيعات)")
        st.plotly_chart(fig1, use_container_width=True)

        # أقل المنتجات مبيعاً
        bottom_sales = df.groupby(col_product)[col_sales].sum().sort_values().head(10)
        fig2 = px.bar(bottom_sales, title="📉 أقل المنتجات مبيعًا")
        st.plotly_chart(fig2, use_container_width=True)


        # تحليل الكمية
        if col_qty:
            qty_rank = df.groupby(col_product)[col_qty].sum().sort_values(ascending=False).head(10)
            fig3 = px.bar(qty_rank, title="📦 أعلى المنتجات في الكمية المباعة")
            st.plotly_chart(fig3, use_container_width=True)

        # تحليل الربح
        if col_profit:
            profit_rank = df.groupby(col_product)[col_profit].sum().sort_values(ascending=False).head(10)
            fig4 = px.bar(profit_rank, title="💰 أكثر المنتجات تحقيقًا للربح")
            st.plotly_chart(fig4, use_container_width=True)

        # تحليل التكلفة
        if col_cost:
            cost_rank = df.groupby(col_product)[col_cost].sum().sort_values(ascending=False).head(10)
            fig5 = px.bar(cost_rank, title="💲 أعلى المنتجات في التكلفة الإجمالية")
            st.plotly_chart(fig5, use_container_width=True)

        # ============================
        # 4) التحليل الزمني
        # ============================
        if col_date:
            st.header("⏳ التحليل الزمني")
            try:
                df[col_date] = pd.to_datetime(df[col_date], errors='coerce')
                time_series = df.groupby(df[col_date].dt.to_period('M'))[col_sales].sum().reset_index()
                time_series[col_date] = time_series[col_date].dt.to_timestamp()
                fig6 = px.line(time_series, x=col_date, y=col_sales, title="📅 المبيعات عبر الزمن")
                st.plotly_chart(fig6, use_container_width=True)
            except:
                st.warning("⚠️ تعذر تنفيذ التحليل الزمني - تأكد من صحة عمود التاريخ.")

        # ============================
        # 5) التنبؤ بالذكاء الاصطناعي Prophet
        # ============================
        if col_date:
            st.header("🤖 التنبؤ بالمبيعات (AI Prophet)")
            try:
                df_prophet = df[[col_date, col_sales]].rename(columns={col_date: "ds", col_sales: "y"})
                df_prophet.dropna(inplace=True)

                model = Prophet()
                model.fit(df_prophet)
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)

                fig7 = plot_plotly(model, forecast)
                st.plotly_chart(fig7)
            except Exception as e:
                st.error(f"❌ خطأ في التنبؤ: {e}")

        # ============================
        # 6) تقرير ذكي من AI
        # ============================
        st.header("🧠 تقرير ذكاء اصطناعي عن حالة المبيعات")

        ai_report = f"""
        🔹 أعلى منتج مبيعًا: {top_sales.index[0]}
        🔹 أعلى منتج في الربح: {profit_rank.index[0] if col_profit else 'غير متوفر'}
        🔹 أعلى منتج في الكمية: {qty_rank.index[0] if col_qty else 'غير متوفر'}
        🔹 اتجاه المبيعات يبدو {'تصاعديًا' if top_sales.iloc[0] > bottom_sales.iloc[0] else 'متذبذبًا'}.
        
        🔍 التوصيات:
        - التركيز على المنتجات الأعلى مبيعًا.
        - تخفيض تكلفة المنتجات الأقل أداءً.
        - دراسة موسمية المبيعات باستخدام التحليل الزمني.
        - استخدام توقعات Prophet لتحسين التخطيط.
        """

        st.success(ai_report)
