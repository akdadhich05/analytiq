import streamlit as st
from sqlalchemy.orm import Session
from models import Dataset, DQRule, get_db
import pandas as pd

def main():
    st.title("Data Quality Rules")
    
    # Fetch datasets from the database
    db: Session = next(get_db())
    datasets = db.query(Dataset).all()
    
    if not datasets:
        st.write("No datasets available. Please upload a dataset first.")
        return
    
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select Dataset", dataset_names)
    
    if selected_dataset_name:
        # Get the selected dataset object
        selected_dataset = db.query(Dataset).filter(Dataset.name == selected_dataset_name).first()
        st.subheader(f"Define Rules for {selected_dataset_name}")
        
        # Load the dataset to get the column names
        df = pd.read_csv(selected_dataset.filepath)
        columns = df.columns.tolist()
        
        # Interface to define a new rule
        with st.form("define_rule"):
            rule_name = st.text_input("Rule Name")
            rule_type = st.selectbox("Rule Type", ["Range Check", "Null Check", "Uniqueness Check", "Custom Lambda"])
            target_columns = st.multiselect("Target Columns", columns)  # Now using multiselect
            condition = st.text_input("Condition (Lambda for Custom Rule)")
            severity = st.selectbox("Severity", ["Warning", "Error"])
            message = st.text_area("Custom Message")
            
            submitted = st.form_submit_button("Add Rule")
            
            if submitted:
                # Save each rule for the selected columns in the database
                for target_column in target_columns:
                    new_rule = DQRule(
                        dataset_id=selected_dataset.id,
                        rule_name=rule_name,
                        rule_type=rule_type,
                        target_column=target_column,
                        condition=condition,
                        severity=severity,
                        message=message
                    )
                    db.add(new_rule)
                db.commit()
                st.success(f"Rule '{rule_name}' added successfully!")
        
        # Display existing rules
        st.subheader(f"Existing Rules for {selected_dataset_name}")
        rules = db.query(DQRule).filter(DQRule.dataset_id == selected_dataset.id).all()
        
        if rules:
            df_rules = pd.DataFrame([{
                "Rule Name": rule.rule_name,
                "Type": rule.rule_type,
                "Column": rule.target_column,
                "Condition": rule.condition,
                "Severity": rule.severity,
                "Message": rule.message
            } for rule in rules])
            st.table(df_rules)
        else:
            st.write("No rules defined for this dataset yet.")

if __name__ == "__main__":
    main()
