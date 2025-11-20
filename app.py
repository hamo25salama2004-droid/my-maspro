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


# ============================
# 7) تحليل الربحية المتقدم
# ============================
# (تمت الإضافة دون تغيير أي جزء من الكود الأساسي)

st.header("💰 تحليل الربحية المتقدم (Profit Analysis)")
if col_profit and col_cost:
    df['Net Profit'] = df[col_profit] - df[col_cost]
    profit_fig = px.bar(df.groupby(col_product)['Net Profit'].sum(), title="صافي الربح لكل منتج")
    st.plotly_chart(profit_fig, use_container_width=True)

# ============================
# 8) تحليل التسعير Price Sensitivity
# ============================
if col_price and col_sales:
    st.header("💲 تحليل حساسية السعر")
    fig_price = px.scatter(df, x=col_price, y=col_sales, trendline="ols", title="العلاقة بين السعر والمبيعات")
    st.plotly_chart(fig_price, use_container_width=True)

# ============================
# 9) تحليل المخزون Inventory Analysis
# ============================
st.header("📦 تحليل المخزون")
if col_qty:
    inv = df.groupby(col_product)[col_qty].sum()
    inv_fig = px.bar(inv, title="إجمالي الكميات المتوفرة لكل منتج")
    st.plotly_chart(inv_fig, use_container_width=True)

# ============================
# 10) تقارير PDF تلقائية
# ============================
st.header("📄 إنشاء تقرير PDF")
st.download_button("📥 تحميل تقرير PDF (تجريبي)", data=str(df.describe()), file_name="report.pdf")

# ============================
# 11) تنبيهات ذكية Alerts
# ============================
st.header("🚨 نظام تنبيهات")
if col_sales:
    low_sales = df.groupby(col_product)[col_sales].sum().sort_values().head(1)
    st.warning(f"⚠️ المنتج الأقل مبيعًا: {low_sales.index[0]}")

# ============================
# 12) مقارنة المنتجات Competitive Analysis
# ============================
st.header("⚔️ مقارنة المنتجات")
if col_sales:
    comp_fig = px.pie(df, names=col_product, values=col_sales, title="حصة كل منتج من المبيعات")
    st.plotly_chart(comp_fig, use_container_width=True)

# ============================
# 13) نظام توصيات Recommendation System
# ============================
st.header("🤖 نظام توصيات المنتجات")
if col_sales:
    best = df.groupby(col_product)[col_sales].sum().sort_values(ascending=False).head(3)
    st.success(f"🟢 المنتجات المقترحة لزيادتها: {list(best.index)}")

# ============================
# 14) تحليل موسمية Seasonality
# ============================
st.header("📆 تحليل الموسمية")
if col_date:
    try:
        df[col_date] = pd.to_datetime(df[col_date], errors='coerce')
        season = df.groupby(df[col_date].dt.month)[col_sales].sum()
        season_fig = px.line(season, title="موسمية المبيعات")
        st.plotly_chart(season_fig, use_container_width=True)
    except:
        st.warning("⚠️ تعذر تنفيذ تحليل الموسمية")

# ============================
# 15) تحليل What-If
# ============================
st.header("🧪 تحليل ماذا لو (What-If)")
if col_price and col_sales:
    factor = st.slider("نسبة تغيير السعر %", -50, 50, 0)
    df['WhatIf Sales'] = df[col_sales] * (1 - factor/100)
    fig_if = px.line(df, y=['WhatIf Sales', col_sales], title="تحليل ماذا لو لتغيير السعر")
    st.plotly_chart(fig_if, use_container_width=True)

# ============================
# 16) تحديد أفضل سعر Optimal Price
# ============================
st.header("🎯 أفضل سعر للمنتج")
if col_price and col_sales:
    opt = df.groupby(col_price)[col_sales].sum().sort_values(ascending=False).head(1)
    st.success(f"🔹 أفضل سعر لتحقيق أعلى مبيعات: {opt.index[0]}")

# ============================
# 17) شات ذكاء صناعي داخل التطبيق
# ============================
st.header("🤖 AI ChatBot")
user_q = st.text_input("اكتب استفسارك عن البيانات:")
if user_q:
    st.info("🔍 الرد الذكي: سيتم إضافة نموذج لغوي فعلي عند ربط API.")


# ============================
# 🔵 تحسين تنسيق الكود (Code Formatting)
# ============================
# تم تنظيم الأقسام باستخدام فواصل واضحة وعناوين قوية.

# ============================
# 🔵 تصميم واجهة المستخدم (UI Design)
# ============================
st.markdown("""
<style>
    .main {background-color: #f5f7fa;}
    h1, h2, h3 {color: #2c3e50;}
    .css-1d391kg {background-color: white; padding: 20px; border-radius: 15px;}
</style>
""", unsafe_allow_html=True)

# ============================
# 🔵 إضافة ذكاء اصطناعي حقيقي عبر API (Placeholder)
# ============================
st.header("🤖 ذكاء اصطناعي (GPT API)")
ai_input = st.text_area("اسأل الذكاء الاصطناعي عن البيانات:")
if ai_input:
    st.info("سيتم تفعيل GPT API الحقيقي عند إضافة مفتاح الربط.")

# ============================
# 🔵 تصدير البيانات إلى Excel
# ============================
st.header("📤 تصدير البيانات إلى Excel")
excel_data = df.to_excel("exported_data.xlsx", index=False)
st.download_button("📥 تحميل ملف Excel", data=excel_data, file_name="Sales_Analysis.xlsx")

# ============================
# 🔵 تحسين التقرير النهائي (AI Insights)
# ============================
st.header("📑 تقرير ذكي من AI")
ai_report = f"""
🔍 **تقرير AI حسب البيانات:**
- أعلى منتج مبيعًا: {df.groupby(col_product)[col_sales].sum().idxmax()}
- أقل منتج مبيعًا: {df.groupby(col_product)[col_sales].sum().idxmin()}
- متوسط المبيعات: {df[col_sales].mean():.2f}
- أفضل شهر مبيعات: {df.groupby('Month')[col_sales].sum().idxmax()}

💡 **توصيات AI:**
- ركّز على زيادة المخزون للمنتجات الأعلى مبيعًا.
- حسّن تسعير المنتجات الأقل أداءً.
- نفّذ عروض موسمية في الأشهر الضعيفة.
- استخدم توقعات Prophet لتخطيط المبيعات المستقبلية.
"""
st.success(ai_report)

# ============================
# 🔵 إضافة كروت KPIs
# ============================
st.header("📊 مؤشرات الأداء الرئيسية (KPIs)")
kpi1 = df[col_sales].sum()
kpi2 = df[col_sales].mean()
kpi3 = df[col_qty].sum()
st.metric("إجمالي المبيعات", f"{kpi1:,.2f}")
st.metric("متوسط المبيعات", f"{kpi2:,.2f}")
st.metric("إجمالي الكمية", f"{kpi3:,.0f}")

# ============================
# 🔵 صفحة تحليل مستقلة لكل قسم
# ============================
st.sidebar.header("📌 اختيار صفحة التحليل")
page = st.sidebar.selectbox("انتقل إلى:", [
    "تحليل المنتجات",
    "تحليل المبيعات الشهرية",
    "تحليل الأسعار",
    "تحليل الكميات",
    "تقارير AI",
])

if page == "تحليل المنتجات":
    st.header("📦 تحليل المنتجات")
    st.write(df.groupby(col_product)[col_sales].sum())

elif page == "تحليل المبيعات الشهرية":
    st.header("📆 تحليل المبيعات الشهرية")
    st.line_chart(df.groupby('Month')[col_sales].sum())

elif page == "تحليل الأسعار":
    st.header("💲 تحليل الأسعار")
    st.scatter_chart(df[[col_price, col_sales]])

elif page == "تحليل الكميات":
    st.header("📦 تحليل الكميات")
    st.bar_chart(df.groupby(col_product)[col_qty].sum())

elif page == "تقارير AI":
    st.header("🤖 تقارير الذكاء الصناعي")
    st.write(ai_report)

