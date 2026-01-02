import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image
st.set_page_config(layout="wide")


def main():

    st.title("Customer Churn Prediction Dashboard")
    st.write("This dashboard predicts whether a customer is likely to churn and highlights churn risk probability.")

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        Gender = st.selectbox("Gender",["Male","Female"],index=None,placeholder="Select Gender")
        Age = st.selectbox( "Age",options=list(range(10,101)),index=None,placeholder="Select Age")

        Country = st.selectbox ('Country' ,['Bangladesh', 'Canada', 'Germany', 'Australia', 'India', 'USA', 'UK'],index=None,placeholder="Select Country")
        City = st.selectbox ('City' ,['London', 'Sydney', 'New York', 'Dhaka', 'Delhi', 'Toronto', 'Berlin'],index=None,placeholder="Select City")

        Customer_Segment = st.selectbox ('Customer Segment' ,['SME', 'Individual', 'Enterprise'],index=None,placeholder="Select Customer Segment")
        Tenure_Months = st.number_input("Tenure (Months)",min_value=0,max_value=1000,step=1,help="How long the customer has been subscribed (in months)")

        Signup_Channel = st.selectbox ('Signup Channel' ,['Web', 'Mobile', 'Referral'],index=None,placeholder="Select Signup Channel")
        Contract_Type = st.selectbox ('Contract Type' ,['Monthly', 'Yearly', 'Quarterly'],index=None,placeholder="Select Contract Type")
    
        Monthly_Logins = st.number_input("Monthly Logins",min_value=0,max_value=1000,step=1,help="How many times the customer logged into the platform in the last 30 days")
        Weekly_Active_Days = st.selectbox( "Weekly Active Days",options=list(range(0,8)),index=None,placeholder="Select Weekly Active Days")

    with col2:
        Avg_Session_Time = st.number_input("Average Session Time (minutes)",min_value=0.0,max_value=1440.0,step=0.5,help="Average time (in minutes) the user spends per session")
        Features_Used = st.selectbox("Features Used",options=list(range(1, 16)),index=None,placeholder="Select Number of Features Used")

        Usage_Growth_Rate = st.number_input("Usage Growth Rate (%)",min_value=-100.0,max_value=100.0,step=1.0,help="Percentage increase or decrease in customer usage compared to last month")
        Last_Login_Days_Ago = st.number_input("Last Login (Days Ago)",min_value=0,max_value=90,step=1,help="Number of days since customer last logged in (0 = logged in today)")

        Monthly_Fee = st.selectbox("Monthly Fee",options=[10, 20, 30, 50, 70, 100],index=None,placeholder="Select Monthly Fee Plan")
        Total_Revenue = st.number_input("Total Revenue Collected ",min_value=0.0,max_value=10000000.0,step=10.0,help="Total amount customer has paid since joining")

        Payment_Method = st.selectbox ('Payment Method' ,['PayPal', 'Card', 'Bank Transfer'],index=None,placeholder="Select Payment Method")
        Payment_Failures = st.number_input("Payment Failures",min_value=0,max_value=10,step=1,help="Number of failed payment attempts by the customer")

        Discount_Applied = st.selectbox ('Discount Applied' ,['Yes', 'No'],index=None,placeholder="Select Discount Applied")
        Price_Increase_Last_3m = st.selectbox ('Price Increase In Last 3 Month ' ,['Yes', 'No'],index=None,placeholder="Select Price Increase In Last 3 Month ")

    with col3:
        Support_Tickets = st.number_input("Support Tickets",min_value=0,max_value=100,step=1,help="Number of support complaints raised by the customer")
        Avg_Resolution_Time = st.number_input("Average Resolution Time (Minute)",min_value=0.0,max_value=14400.0,step=1.0,help="Average number of hours taken to resolve customer issues")

        Complaint_Type = st.selectbox ('Complaint Type' ,['Service', 'Billing' ,'Technical','No_Complaint'],index=None,placeholder="Select Complaint Type")
        Csat_Score = st.selectbox("CSAT Score (1 = Very Bad, 5 = Excellent)",[1, 2, 3, 4, 5],index=None,placeholder="Select CSAT Score")

        Escalations = st.number_input("Escalations",min_value=0,max_value=4,step=1,help="Number of times the customer's issue was escalated to higher-level support")
        Email_Open_Rate = st.number_input("Email Open Rate (%)",min_value=0.0,max_value=100.0,step=1.0,help="Percentage of marketing emails opened by the customer")

        Marketing_Click_Rate = st.number_input("Marketing Click Rate (%)",min_value=0.0,max_value=100.0,step=1.0,help="Percentage of marketing emails/messages that the customer clicked")
        nps_Score = st.number_input("NPS Score",min_value=-100,max_value=100,step=1,help="Net Promoter Score ranges from -100 (very unhappy) to +100 (highly loyal)")

        Survey_Response = st.selectbox ('Survey Response' ,['Satisfied','Neutral', 'Unsatisfied'],index=None,placeholder="Select Survey Response")
        Referral_Count = st.number_input("Referral Count",min_value=0,max_value=1000,step=1,help="Number of customers referred by this user")


    model = pickle.load(open('model.sav','rb'))

    scaler = pickle.load (open ('scalar.sav' , 'rb'))

    gender_obj = pickle.load(open('one_gender.sav' ,'rb'))
    country_obj = pickle.load(open('one_country.sav' ,'rb'))
    city_obj = pickle.load (open('one_city.sav' , 'rb'))
    customer_segment_obj = pickle.load (open ('one_customer_segment.sav' , 'rb'))

    signup_channel_obj = pickle.load (open ('one_signup_channel.sav' , 'rb'))
    contract_type_obj = pickle.load (open ('one_contract_type.sav' , 'rb'))
    payment_method_obj = pickle.load (open ('one_payment_method.sav' , 'rb'))
    discount_applied_obj = pickle.load (open ('one_discount_applied.sav' , 'rb'))

    price_increase_obj = pickle.load (open ('one_price_increase_last_3m.sav' , 'rb'))
    complaint_type_obj = pickle.load (open ('one_complaint_type.sav' , 'rb'))
    survey_response_obj = pickle.load (open ('one_survey_response.sav' , 'rb'))

    pred = st.button('PREDICT')
    print(pred)

    def generate_recommendations(
        Monthly_Logins, Weekly_Active_Days, Avg_Session_Time, 
        Features_Used, Usage_Growth_Rate, Last_Login_Days_Ago,
        Monthly_Fee, Payment_Failures, Discount_Applied, Price_Increase_Last_3m,
        Support_Tickets, Avg_Resolution_Time, Complaint_Type,
        Csat_Score, Escalations, Email_Open_Rate, Marketing_Click_Rate,
        nps_Score, Survey_Response, Referral_Count):
    
        recommendations = []

        # Engagement & Usage Recommendations

        if int(Monthly_Logins) < 10 or int(Weekly_Active_Days) <= 1:
            recommendations.append("Increase user engagement through reminders, push notifications, and personalized emails.")

        if float(Usage_Growth_Rate) < 0:
            recommendations.append("User activity is declining. Launch re-engagement campaigns with offers or content highlights.")

        if float(Avg_Session_Time) < 10:
            recommendations.append("Customer is not spending enough time on platform. Provide tutorials or feature onboarding.")

        if int(Features_Used) < 5:
            recommendations.append("Customer is not exploring product features. Recommend feature education or guided walkthrough.")

        if int(Last_Login_Days_Ago) > 10:
            recommendations.append("Customer has been inactive recently — trigger comeback notification or incentive.")

        # Billing Recommendations
        if int(Payment_Failures) > 0:
            recommendations.append("Payment failures detected. Suggest alternate payment methods or billing support.")

        if Discount_Applied == "No" and float(Usage_Growth_Rate) < 0:
            recommendations.append("Consider providing loyalty discounts or retention offers.")

        if Price_Increase_Last_3m == "Yes":
            recommendations.append("Customer experienced recent price hike — provide compensation benefits or reassurance messaging.")

        # Support & Complaint Recommendations
        if int(Support_Tickets) > 2:
            recommendations.append("High support ticket count — assign priority support or dedicated relationship manager.")

        if float(Avg_Resolution_Time) > 24:
            recommendations.append("Slow issue resolution detected — improve service speed to prevent dissatisfaction.")

        if Complaint_Type != "No_Complaint":
            recommendations.append("Customer had complaints — conduct follow-up to ensure satisfaction recovery.")

        if int(Escalations) > 0:
            recommendations.append("Escalations recorded — provide apology communication and service assurance.")

        # Satisfaction & Loyalty Recommendations
        if int(Csat_Score) <= 2:
            recommendations.append("Very low CSAT — initiate immediate feedback call and corrective action plan.")

        if nps_Score < 0:
            recommendations.append("Negative NPS — customer is unhappy. Deploy retention recovery strategy.")

        if Survey_Response == "Unsatisfied":
            recommendations.append("Customer reported dissatisfaction — request feedback and improve service experience.")

        # Marketing & Communication
        if float(Email_Open_Rate) < 20:
            recommendations.append("Low email engagement — try SMS, push notifications, or personalized campaigns.")

        if float(Marketing_Click_Rate) < 10:
            recommendations.append("Customer is not engaging with marketing — optimize targeting and messaging.")

        # Referral / Loyalty
        if int(Referral_Count) == 0 and nps_Score > 30:
            recommendations.append("Customer is satisfied but not referring — introduce referral rewards or loyalty program.")

        # Default Safe Suggestion
        if len(recommendations) == 0:
            recommendations.append("Customer is healthy. Maintain engagement and continue monitoring behavior.")

        return recommendations

    if pred :

        gender = gender_obj.transform ([[Gender]])
        country = country_obj.transform ([[Country]])
        city = city_obj.transform ([[City]])
        customer_segment = customer_segment_obj.transform ([[Customer_Segment]])

        signup_channel = signup_channel_obj.transform ([[Signup_Channel]])
        contract_type = contract_type_obj.transform ([[Contract_Type]])
        payment_method = payment_method_obj.transform ([[Payment_Method]])
        discount_applied = discount_applied_obj.transform ([[Discount_Applied]])

        price_increase = price_increase_obj.transform ([[Price_Increase_Last_3m]])
        complaint_type = complaint_type_obj.transform ([[Complaint_Type]])
        survey_response = survey_response_obj.transform ([[Survey_Response]])

        data = np.array ([Age, Tenure_Months, Monthly_Logins, Weekly_Active_Days, Avg_Session_Time , 
                  Features_Used, Usage_Growth_Rate, Last_Login_Days_Ago, Monthly_Fee , Total_Revenue,
                  Payment_Failures, Support_Tickets, Avg_Resolution_Time , Csat_Score , Escalations,
                  Email_Open_Rate, Marketing_Click_Rate, nps_Score, Referral_Count])

        data2 = np.concatenate([gender,country],axis = 1)
        data3 = np.concatenate([data2,city],axis = 1)
        data4 = np.concatenate([data3,customer_segment],axis = 1)
        data5 = np.concatenate([data4,signup_channel],axis = 1)

        data6 = np.concatenate([data5,contract_type],axis = 1)
        data7 = np.concatenate([data6,payment_method],axis = 1)
        data8 = np.concatenate([data7,discount_applied],axis = 1)
        data9 = np.concatenate([data8,price_increase],axis = 1)

        data10 = np.concatenate([data9,complaint_type],axis = 1)
        data11 = np.concatenate([data10,survey_response],axis = 1)

        data12 = data.reshape(1,-1)

        scaler_data = scaler.transform(data12)

        prediction_data = np.concatenate([scaler_data,data11],axis = 1)

        predicted_output = model.predict(prediction_data)

        label = predicted_output[0]

        if label == 1:
            st.error(" Customer is LIKELY TO LEAVE (Churn)")
        else:
            st.success(" Customer is NOT LIKELY TO LEAVE")

        st.subheader("Recommendations to Improve Customer Retention")

        recommendations = generate_recommendations(Monthly_Logins, Weekly_Active_Days, Avg_Session_Time,
                                                    Features_Used, Usage_Growth_Rate, Last_Login_Days_Ago,
                                                    Monthly_Fee, Payment_Failures, Discount_Applied, Price_Increase_Last_3m,
                                                    Support_Tickets, Avg_Resolution_Time, Complaint_Type,
                                                    Csat_Score, Escalations, Email_Open_Rate, Marketing_Click_Rate,
                                                    nps_Score, Survey_Response, Referral_Count)

        for r in recommendations:
            st.write("✔️", r)

        

main()
