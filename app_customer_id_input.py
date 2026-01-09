import pandas as pd
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")

df = pd.read_csv("customer_churn_business_dataset.csv")


def main():


    st.title("Customer Churn Prediction Dashboard")
    st.write("This dashboard predicts whether a customer is likely to churn and highlights churn risk probability.")

    image = Image.open('image.png')
    st.image(image,width=700)

    st.subheader("Fetch Customer Details")

    customer_id = st.text_input("Enter Customer ID")
    run_btn = st.button("Fetch & Predict")

    if run_btn:

        customer_row = df[df["customer_id"] == customer_id]

        if customer_row.empty:
            st.error(" Customer ID not found")
            st.stop()

        st.success(" Customer Found")

        
        customer_row = customer_row.drop(columns=["customer_id","churn"])


        Gender = customer_row["gender"].values[0]
        Country = customer_row["country"].values[0]
        City = customer_row["city"].values[0]
        Customer_Segment = customer_row["customer_segment"].values[0]

        Age = int(customer_row["age"].values[0])
        Tenure_Months = int(customer_row["tenure_months"].values[0])
        Monthly_Logins = int(customer_row["monthly_logins"].values[0])
        Weekly_Active_Days = int(customer_row["weekly_active_days"].values[0])
        Avg_Session_Time = float(customer_row["avg_session_time"].values[0])

        Features_Used = int(customer_row["features_used"].values[0])
        Usage_Growth_Rate = float(customer_row["usage_growth_rate"].values[0])
        Last_Login_Days_Ago = int(customer_row["last_login_days_ago"].values[0])
        Monthly_Fee = int(customer_row["monthly_fee"].values[0])
        Total_Revenue = float(customer_row["total_revenue"].values[0])

        Payment_Failures = int(customer_row["payment_failures"].values[0])
        Support_Tickets = int(customer_row["support_tickets"].values[0])
        Avg_Resolution_Time = float(customer_row["avg_resolution_time"].values[0])
        Csat_Score = int(customer_row["csat_score"].values[0])
        Escalations = int(customer_row["escalations"].values[0])

        Email_Open_Rate = float(customer_row["email_open_rate"].values[0])
        Marketing_Click_Rate = float(customer_row["marketing_click_rate"].values[0])
        nps_Score = int(customer_row["nps_score"].values[0])
        Referral_Count = int(customer_row["referral_count"].values[0])

        Signup_Channel = customer_row["signup_channel"].values[0]
        Contract_Type = customer_row["contract_type"].values[0]
        Payment_Method = customer_row["payment_method"].values[0]
        Discount_Applied = customer_row["discount_applied"].values[0]
        Price_Increase_Last_3m = customer_row["price_increase_last_3m"].values[0]
        Complaint_Type = customer_row["complaint_type"].values[0]
        Survey_Response = customer_row["survey_response"].values[0]

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
