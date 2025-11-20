# Full Sales Data Analysis & Dashboard Code
# Full Streamlit Sales Analysis App + Advanced AI Processing
# يشمل: تحميل – تنظيف – تحليل – ذكاء اصطناعي – Dashboard كاملة
# النظام مناسب للمؤسسات الكبيرة

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import re

st.set_page_config(page_title="Enterprise Sales Analytics AI System", layout="wide")
st.title("🤖 نظام تحليل بيانات المبيعات المتكامل بالذكاء الاصطناعي")

st.sidebar.header("📂 تحميل الملف")
file = st.sidebar.file_uploader("ارفع ملف Excel أو CSV", type=["xlsx", "xls", "csv"])

#########################################
# AI Helper – يقوم بالفهم التلقائي للعمود
#########################################
def ai_detect_column(df, keywords):
    for col in df.columns:
        for k in keywords:
            if re.search(k, col, re.IGNORECASE):
                return col
    return None

if file:
    # قراءة الملف
    if file.name.endswith("csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("🔍 أول 20 صف في البيانات")
    st.dataframe(df.head(20))

    st.subheader("📌 أنواع الأعمدة")
    st.write(df.dtypes)

    ###############################################
    # AI: تحديد الأعمدة تلقائياً
    ###############################################
    st.sidebar.header("🤖 AI Column Detection")
    date_col = ai_detect_column(df, ["date", "تاريخ"])
    product_col = ai_detect_column(df, ["product", "المنتج"])
    qty_col = ai_detect_column(df, ["qty", "quantity", "الكمية"])
    price_col = ai_detect_column(df, ["price", "السعر"])
    total_col = ai_detect_column(df, ["total", "اجمالي", "إجمالي"])

    # إدخال يدوي عند الحاجة
    date_col = st.sidebar.text_input("اسم عمود التاريخ", value=date_col or "")
    product_col = st.sidebar.text_input("اسم عمود المنتج", value=product_col or "")
    qty_col = st.sidebar.text_input("اسم عمود الكمية", value=qty_col or "")
    price_col = st.sidebar.text_input("اسم عمود السعر", value=price_col or "")
    total_col = st.sidebar.text_input("اسم عمود إجمالي المبيعات", value=total_col or "")

    #########################################################
    # تنظيف كامل للبيانات كما في الشركات الكبيرة
    #########################################################
    st.header("🧹 تنظيف البيانات – مستوى شركات")

    # إزالة الصفوف المكررة
    df.drop_duplicates(inplace=True)

    # معالجة القيم المفقودة
    imputer = SimpleImputer(strategy="median")
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # قيم مفقودة للنوعي
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    # اكتشاف القيم الشاذة
    if qty_col:
        iso = IsolationForest(contamination=0.02)
        df['anomaly'] = iso.fit_predict(df[[qty_col]])
        df = df[df['anomaly'] == 1]
        df.drop(columns=['anomaly'], inplace=True)

    st.success("✔️ تم تنظيف البيانات بالكامل")

    #########################################################
    # تجهيز بيانات المبيعات
    #########################################################
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['Month'] = df[date_col].dt.to_period('M').astype(str)

    if total_col == "" and price_col != "" and qty_col != "":
        df['Total'] = df[price_col] * df[qty_col]
        total_col = 'Total'

    #########################################################
    # جميع التحليلات الكاملة
    #########################################################
    st.header("📊 جميع التحليلات المتقدمة")

    # 1 – أعلى منتج مبيعًا
    st.subheader("🔥 أعلى منتج مبيعًا")
    st.write(df.groupby(product_col)[total_col].sum().sort_values(ascending=False).head(5))

    # 2 – أقل منتج مبيعًا
    st.subheader("❄️ أقل المنتجات مبيعًا")
    st.write(df.groupby(product_col)[total_col].sum().sort_values().head(5))

    # 3 – المبيعات الشهرية
    st.subheader("📆 المبيعات الشهرية")
    monthly = df.groupby('Month')[total_col].sum()
    st.line_chart(monthly)

    # 4 – تحليل العملاء (لو موجود عمود عميل)
    customer_cols = [c for c in df.columns if re.search("customer|عميل", c, re.IGNORECASE)]
    if customer_cols:
        cust = customer_cols[0]
        st.subheader("🧍‍♂️ تحليل العملاء")
        st.write(df.groupby(cust)[total_col].sum().sort_values(ascending=False).head(10))

    # 5 – تحليل الفئات إن وجدت
    st.subheader("📦 تحليل المنتجات")
    prod_sales = df.groupby(product_col)[total_col].sum().sort_values(ascending=False)
    st.plotly_chart(px.bar(prod_sales, title="إجمالي المبيعات لكل منتج"), use_container_width=True)

    #########################################################
    # AI-based Clustering (لتقسيم العملاء/المنتجات)
    #########################################################
    st.header("🤖 تحليل الذكاء الاصطناعي – التجميع (Clustering)")

    try:
        scale_cols = [qty_col, price_col, total_col]
        scaler = StandardScaler()
        X = scaler.fit_transform(df[scale_cols])
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)
        st.write(df[['Cluster'] + scale_cols].head())
        st.plotly_chart(px.scatter(df, x=qty_col, y=total_col, color='Cluster', title="AI Clustering"))
    except:
        st.warning("تعذر تنفيذ التجميع – قد تكون البيانات غير مناسبة")

    #########################################################
    # توقع المبيعات Prophet
    #########################################################
    st.header("🔮 التنبؤ بالمبيعات (Prophet)")
    try:
        forecast_df = df.groupby(date_col)[total_col].sum().reset_index()
        forecast_df.columns = ['ds', 'y']
        model = Prophet()
        model.fit(forecast_df)
        future = model.make_future_dataframe(periods=60)
        forecast = model.predict(future)
        st.plotly_chart(px.line(forecast, x='ds', y='yhat', title='توقع المبيعات 60 يوم'))
    except:
        st.warning("تعذر إجراء التنبؤ – تأكد من وجود عمود تاريخ صالح")

    st.success("🎯 النظام جاهز – جميع التحليلات تمت بنجاح + ذكاء اصطناعي + تنظيف مؤسسي")
# يقبل ملفات عربية وإنجليزية + جميع التحليلات + Dashboard كاملة

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet

st.set_page_config(page_title="Sales Analysis Dashboard", layout="wide")
st.title("📊 نظام تحليل بيانات المبيعات الكامل")

st.sidebar.header("📂 تحميل الملف")
file = st.sidebar.file_uploader("ارفع ملف Excel أو CSV", type=["xlsx", "xls", "csv"])

if file:
    # قراءة الملف
    if file.name.endswith("csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("عرض أول 20 صف")
    st.dataframe(df.head(20))

    st.subheader("معلومات عن الأعمدة")
    st.write(df.dtypes)

    st.sidebar.header("⚙️ تحديد الأعمدة للتحليل")
    date_col = st.sidebar.text_input("اكتب اسم عمود التاريخ كما هو في الملف")
    product_col = st.sidebar.text_input("اكتب اسم عمود اسم المنتج")
    qty_col = st.sidebar.text_input("اكتب اسم عمود الكمية")
    price_col = st.sidebar.text_input("اكتب اسم عمود السعر")
    total_col = st.sidebar.text_input("اكتب اسم عمود إجمالي المبيعات")

    if date_col and product_col and qty_col and price_col:
        # معالجة التاريخ
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['Month'] = df[date_col].dt.to_period('M').astype(str)

        # حساب إجمالي المبيعات لو مش موجود
        if total_col == "":
            df['Total'] = df[price_col] * df[qty_col]
            total_col = 'Total'

        st.header("📈 التحليلات الأساسية")

        # أعلى منتج مبيعا
        best_product = df.groupby(product_col)[total_col].sum().sort_values(ascending=False).head(1)
        st.subheader("🔥 أعلى منتج مبيعًا")
        st.write(best_product)

        # أقل منتج مبيعاً
        st.subheader("❄️ أقل منتج مبيعًا")
        st.write(df.groupby(product_col)[total_col].sum().sort_values().head(1))

        # مبيعات شهرية
        monthly = df.groupby('Month')[total_col].sum()
        st.subheader("📆 المبيعات الشهرية")
        st.line_chart(monthly)

        # تحليل الفئات لو موجود
        st.subheader("📦 تحليل المنتجات")
        product_sales = df.groupby(product_col)[total_col].sum().sort_values(ascending=False)
        fig = px.bar(product_sales, title="إجمالي المبيعات لكل منتج")
        st.plotly_chart(fig, use_container_width=True)

        # Prophet التنبؤ
        st.header("🔮 التنبؤ بالمبيعات (Prophet)")
        forecast_df = df.groupby(date_col)[total_col].sum().reset_index()
        forecast_df.columns = ['ds', 'y']
        model = Prophet()
        model.fit(forecast_df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        st.write(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail())
        fig2 = px.line(forecast, x='ds', y='yhat', title='توقع المبيعات')
        st.plotly_chart(fig2, use_container_width=True)

        st.success("✔️ التحليل مكتمل بنجاح – التطبيق جاهز بالكامل!")
