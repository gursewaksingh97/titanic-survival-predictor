#basic Structure
import streamlit as st 
import pandas as pd
from joblib import load
#joblib is used to save and load ml models 

# load model  and columns
model=load('tree.joblib')
train_columns =load('columns.joblib')
data=load('data.joblib') #this is x

# Page Setting 
st.set_page_config(page_title="Titanic Predictions Interface",layout="wide")

st.title("🚢 Titanic Dashboard")
st.markdown("### *Explore passenger data + Predict Survival*")
st.divider()
# ------------------------------------------------------------------------------------------------
#sidebar
st.sidebar.header("🎯 Passengers Details")
Pclass=st.sidebar.radio("Passsanger Class",[1,2,3])

Sex= st.sidebar.radio("Sex",["Male","Female"])
Sex=1 if Sex=="Male" else 0

Embarked=st.sidebar.selectbox("Embarked",['Cherbourge','Queenstown','Southampton'])
embarked_map={'Cherbourge':0,'Queenstown':1,'Southampton':2}
Embarked=embarked_map[Embarked]

Age_cat=st.sidebar.slider("Age Category",0,3,1)
Fare_cat=st.sidebar.slider("Fare Category",0,3,1)

Family=st.sidebar.radio("Family Onboards?",["No","Yes"])
Family = 1 if Family=='Yes' else 0

inputdf=pd.DataFrame({
    'Passenger Class':[Pclass],
    'Sex':[Sex],
    'Embarked':[Embarked],
    'Age Category':Age_cat,
    'Fare Category':Fare_cat,
    'Family':[Family]
})

# as model expects data in table form i.e Dataframe
# here in streamlitwe pass/gives individual values
# So we convert inputs to a dataframe
# and ,as DataFrame needs Rows , thus we have utilized list[]

# Match Training Columns
inputdf=inputdf.reindex(columns=train_columns,fill_value=0)
# this ensures same columns,Same order and no missing Values
# --------------------------------


# Dataset Insights
st.subheader("📊 Dataset Insights")

col1,col2,col3=st.columns(3)

with col1:
    st.metric("Total Passangers",len(data))
    st.bar_chart(data['Pclass'].value_counts())

with col2:
    male_count=int((data['Sex']==1).sum())
    female_count=int((data['Sex']==0).sum())
    g1,g2=st.columns(2)
    with g1:
        st.metric("Female",female_count)
    with g2:
        st.metric("Male",male_count)
    st.bar_chart(data['Sex'].value_counts())

with col3:
    st.metric("Unique Embarked",data['Embarked'].nunique())
    st.bar_chart(data['Embarked'].value_counts())

st.divider()

# ----====-=-=--=-=-==-=-=-=-=-=-=-=-[]
st.subheader("🔮 Prediction Panel")
pred_prob_col,reset_col=st.columns(2)
with pred_prob_col:
    if st.button("Predict"):
        result=model.predict(inputdf)
        prob=model.predict_proba(inputdf)
    
        if result[0]==1:
            st.success("Survived ✅ ")
        else:
            st.error("Did Not Survived ❌")

        st.session_state['prob']=prob
        st.session_state['inputdf']=inputdf
        st.metric("Survival Probability: ",f"{round(st.session_state['prob'][0][1],2)*100}%")
with reset_col:
    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()
st.divider()   
# ----------------------------input summary----------------------------------------------------------
st.subheader("🧾 Input Summary")
st.write(inputdf)
