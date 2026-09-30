# app.py
import streamlit as st
from recommend import craft_recommendation
st.set_page_config(page_title="HairSense Hybrid", layout="centered")

st.title("💇 HairSense — Personalized Haircare & Lifestyle Recommender")

st.markdown("Fill the form below. Use the multi-select for hair problems.")

with st.form("form"):
    age = st.number_input("Age", 10, 80, 25)
    gender = st.selectbox("Gender", ["female", "male", "other"])
    location = st.text_input("Location (optional)", "")
    wash_freq = st.selectbox("How often do you wash your hair per week?",
                             ["daily", "1–2 times", "2–3 times", "weekly", "unknown"])
    texture = st.selectbox("Natural hair texture",
                           ["straight", "wavy", "curly", "coily", "unknown"])
    density = st.selectbox("Hair density", ["thin", "medium", "thick", "unknown"])
    porosity = st.selectbox("Hair porosity", ["low", "normal", "high", "unknown"])
    scalp_type = st.selectbox("Scalp Type", ["oily", "dry", "normal", "combination", "unknown"])
    hair_problems = st.multiselect("Select hair problems (multi)",
                                   ["dandruff", "hair fall", "dry hair", "oily scalp", "frizzy hair",
                                    "split ends", "itchy scalp", "thin hair"])
    stress = st.selectbox("Daily stress level", ["low", "medium", "high", "unknown"])
    sleep_hours = st.number_input("Average sleep hours/day", 0.0, 12.0, 7.0, 0.5)
    water_liters = st.number_input("Water liters/day", 0.0, 10.0, 2.0, 0.1)
    diet = st.selectbox("Diet", ["mixed", "veg", "non-veg", "unknown"])

    submitted = st.form_submit_button("Get Recommendations")

if submitted:
    user = {
        "age": age,
        "gender": gender,
        "location": location,
        "wash_freq": wash_freq,
        "texture": texture,
        "density": density,
        "porosity": porosity,
        "scalp_type": scalp_type,
        "hair_concerns": ",".join(hair_problems),
        "hair_problems": hair_problems,
        "stress": stress,
        "sleep_hours": sleep_hours,
        "water_liters": water_liters,
        "diet": diet
    }

    res = craft_recommendation(user)

    st.subheader("Assigned cluster")
    st.write(res["cluster"])

    st.subheader("Shampoo suggestions")
    for s in res["shampoo_suggestion"]:
        st.write("•", s)

    st.subheader("Hair masks")
    for s in res["hair_masks"]:
        st.write("•", s)

    st.subheader("Home remedies")
    for s in res["home_remedies"]:
        st.write("•", s)

    st.subheader("Supplements")
    for s in res["supplements"]:
        st.write("•", s)

    st.subheader("Dos")
    for s in res["dos"]:
        st.write("•", s)

    st.subheader("Don'ts")
    for s in res["donts"]:
        st.write("•", s)

    st.subheader("Tips & Diet")
    for s in res["recommendation_tips"]:
        st.write("•", s)
    for s in res["diet_tips"]:
        st.write("•", s)

 #   st.subheader("Cluster profile (brief)")
  #  st.json(res["cluster_profile"], expanded=False)
