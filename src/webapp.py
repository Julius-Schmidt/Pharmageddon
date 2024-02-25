import torch
import streamlit as st
import polars as pl
import pandas as pd


# get model arcitecture


# back end
def main():
    @st.cache_resource
    def load_graph(graph_path):
        graph = torch.load(graph_path)
        return graph

    @st.cache_resource
    def load_model(checkpoint_path):
        model = torch.load(checkpoint_path)
        return model

    @st.cache_data
    def load_data():
        df_drugs = pl.read_csv(DRUGS_PATH).to_pandas()
        df_side_effects = pl.read_csv(SIDE_EFFECTS_PATH).to_pandas()
        return df_drugs, df_side_effects

    @st.cache_data
    def get_select_options(
        df_drugs, df_side_effects, drug_option_column, side_effect_option_column
    ):
        # List of drug and side effect options
        drug_options = df_drugs[
            drug_option_column
        ]  # .unique(subset=["drug_concept_name"], maintain_order=True)["drug_concept_name"]
        side_effect_options = df_side_effects[
            side_effect_option_column
        ]  # ['Dizziness', 'Nausea']
        return drug_options, side_effect_options

    @st.cache_data
    def get_mapping_dict(df_drugs):
        drug_name_to_chemical = pd.Series(
            df_drugs.chemical.values, index=df_drugs.drug_concept_name
        ).to_dict()
        chemical_to_drug_name = pd.Series(
            df_drugs.drug_concept_name.values, index=df_drugs.chemical
        ).to_dict()
        return drug_name_to_chemical, chemical_to_drug_name

    @st.cache_data
    def load_database():
        data = pl.read_csv(DATABASE_PATH)
        return data

    def predict(model, drugs, effects=None):
        if len(effects) == 0:
            effects = graph["effects"].values.tolist()
            effect_ids = torch.tensor(graph["effects"].index)
        else:
            # updated to have not sorted indices
            effect_indices = [
                graph["effects"][graph["effects"] == effect].index[0]
                for effect in effects
            ]
            effect_ids = torch.tensor(effect_indices)

        drug_ids = torch.tensor(
            graph["drugs"]["label"][graph["drugs"]["label"].isin(drugs)].index
        )
        drug_ids = [drug_ids for _ in range(len(effect_ids))]
        # output a dataframe
        with torch.no_grad():
            res = model(graph, drug_ids, effect_ids)
            # Create a list to hold the data
            data = []
            for r, e in zip(res, effects):
                data.append({"Effect": e, "Confident_Score": round(float(r), 3)})

            result_df = pd.DataFrame(data)
            return result_df

    def run_pharmageddon_predict(model, chemicals, effects, chemical_to_drug_name):
        # Build the command string
        result_df = predict(model, chemicals, effects)
        return result_df

    def show_disclaimer():
        st.warning(
            """
                    Disclaimer: Educational Purpose Only\n
                    We've designed this platform as a proof of concept to showcase innovative ideas and potential advancements in the field of healthcare. Our goal is to stimulate discussion, inspire further research, and share knowledge with enthusiasts, professionals, and students alike.\n
                    **Important Notice:**\n
                    The content provided here is for informational purposes only and is not intended as medical advice, diagnosis, or treatment. While we strive to share accurate and up-to-date information, our platform is not a substitute for professional medical guidance. Always seek the advice of your physician or another qualified health provider with any questions you may have regarding a medical condition or treatment options.\n
                    Before making any changes to your healthcare regimen, including the introduction or discontinuation of medications, it is crucial to consult a healthcare professional. Self-diagnosing and self-medicating can be dangerous and lead to unintended consequences.\n
                    If you choose to use any information on this website for educational purposes or as a basis for further research, do so responsibly and critically. Evaluate the information in the context of professional guidelines and in consultation with healthcare providers.\n
                    The creators of this website shall not be liable for any direct, indirect, incidental, consequential, or punitive damages arising from the use of, or the inability to use, the site or its content. Users assume full responsibility and risk for the use of this site and any site-related services.\n
                    **Our Commitment:**\n
                    We are committed to providing a platform that encourages learning and innovation. However, we recognize the importance of making informed decisions, especially when it comes to health and well-being. We urge all visitors to use this website as a springboard for ideas and conversations with healthcare professionals, not as a final authority on medical issues.\n
                    Thank you for visiting our website. We hope it serves as a valuable resource in your quest for knowledge and innovation in healthcare.
                    """,
            icon="⚠️",
        )

    disclaimer_accepted = st.sidebar.checkbox("I accept the disclaimer")

    if disclaimer_accepted:
        # for model predictions
        BASIC_PATH = "./data/"
        DRUGS_PATH = BASIC_PATH + "drug_stitch_table.csv"
        SIDE_EFFECTS_PATH = BASIC_PATH + "side_effects.csv"
        GRAPH_PATH = BASIC_PATH + "out/graph.pk"
        CHECKPOINT_PATH = (
            BASIC_PATH + "out/model/model_0_800.pt"
        )  # "D:\d_drive\Work thingie\hiwi\louis\database\src\streamline\out_test\model\model_0_10.pt"
        DATABASE_PATH = BASIC_PATH + "sorted_poly.csv"

        # load data
        df_drugs, df_side_effects = load_data()

        # load database
        database = load_database()
        # List of drug and side effect options and create a mapping from drug concept name to chemical
        drug_options, side_effects_options = get_select_options(
            df_drugs=df_drugs,
            df_side_effects=df_side_effects,
            drug_option_column="drug_concept_name",
            side_effect_option_column="condition_concept_name",
        )

        # Create a mapping from drug concept name to chemical
        drug_name_to_chemical, chemical_to_drug_name = get_mapping_dict(
            df_drugs=df_drugs
        )

        # load graph
        graph = load_graph(GRAPH_PATH)

        # load PHARMAGEDDON
        model = load_model(CHECKPOINT_PATH)

        # Front end

        # load svg file
        # with open("D:\\d_drive\\Work thingie\\hiwi\\louis\\database\\src\\streamlit\\logo\\pilly.svg", "r") as file:
        #    svg_logo = file.read()
        # st.image(svg_logo, width=10)
        # image = Image.open('sunrise.jpg')

        # st.image(image, caption='Sunrise by the mountains')
        function_options = ("Model Prediction", "Database Search")
        selected_function = st.sidebar.selectbox(
            "Polypharmacy Search", function_options
        )

        if selected_function == "Model Prediction":
            st.title("Pharmageddon")
            st.subheader("Polypharmacy Side Effect Predictor")
            # st.balloons()

            # Get input from user
            session_state = st.session_state
            session_state["drugs_input"] = []

            drugs_input = st.multiselect(
                "Select Drugs:",
                options=drug_options,
                default=session_state["drugs_input"],
            )
            session_state["drugs_input"] = drugs_input

            side_effects_input = st.multiselect(
                "Select Side Effects:", options=side_effects_options
            )

            # TODO: database search mark

            # plug in model here
            if st.button("Submit"):
                # Convert the user inputs into lists of strings

                chemicals = [drug_name_to_chemical[drug] for drug in drugs_input]

                side_effects_list = [str(effect) for effect in side_effects_input]

                # Call the run_pharmageddon_predict function with the selected drugs and side effects
                prediction_output = run_pharmageddon_predict(
                    model=model,
                    chemicals=chemicals,
                    effects=side_effects_list,
                    chemical_to_drug_name=chemical_to_drug_name,
                )
                st.session_state["prediction_output"] = prediction_output
                # add table title
                # st.dataframe(prediction_output)

            # Boolean to resize the dataframe, stored as a session state variable
            # st.checkbox("Use container width", value=False, key="use_container_width")
            st.session_state.use_container_width = True

            # Display the dataframe continuously
            if "prediction_output" in st.session_state:
                st.subheader("Prediction Table")
                prediction_output = st.session_state["prediction_output"].copy()
                prediction_output = prediction_output.sort_values(by="Confident_Score", ascending=False)
                # convert confident score to string
                prediction_output["Confident_Score"] = prediction_output[
                    "Confident_Score"
                ].astype(str)
                st.dataframe(
                    prediction_output,
                    use_container_width=st.session_state.use_container_width,
                )
                csv = prediction_output.to_csv(index=False)
                st.download_button(
                    label="Download CSV File",
                    data=csv,
                    file_name="output.csv",
                    mime="text/csv",
                )
            # Add a slider for the threshold value
            threshold = st.slider(
                "Select Threshold:", min_value=0.0, max_value=1.0, value=0.5, step=0.01
            )

            # Add a button to apply the threshold
            if st.button("Apply Threshold"):
                filtered_output = st.session_state["prediction_output"][
                    st.session_state["prediction_output"]["Confident_Score"]
                    >= threshold
                ].copy()
                st.session_state["filtered_output"] = filtered_output

            # Display the dataframe continuously
            if "filtered_output" in st.session_state:
                st.subheader("Prediction Table with Threshold")
                filtered_output = st.session_state["filtered_output"]
                filtered_output["Confident_Score"] = filtered_output[
                    "Confident_Score"
                ].astype(str)
                filtered_output = filtered_output.sort_values(by="Confident_Score", ascending=False)
                st.dataframe(
                    filtered_output,
                    use_container_width=st.session_state.use_container_width,
                    
                )
                csv = filtered_output.to_csv(index=False)
                st.download_button(
                    label="Download CSV File",
                    data=csv,
                    file_name="filtered_output.csv",
                    mime="text/csv",
                )

        if selected_function == "Database Search":
            st.title("Polypharmacy Side Effect Database")
            # Get input from user
            session_state = st.session_state
            session_state["drugs_input"] = []

            drugs_input = st.multiselect(
                "Select Drugs:",
                options=drug_options,
                default=session_state["drugs_input"],
            )
            session_state["drugs_input"] = drugs_input
            # Boolean to resize the dataframe, stored as a session state variable

            if st.button("Submit"):
                # Convert the user inputs into lists of strings
                # Apply function to data
                chemicals = [drug_name_to_chemical[drug] for drug in drugs_input]

                # Search for drugs in database

                # Filter data
                # Step 2: Sort your list of chemicals and join them into a string with '|' as the delimiter
                sorted_chemicals_str = "|".join(
                    sorted(chemicals, key=lambda x: int(x[3:]))
                )

                # Step 3: Filter the DataFrame for rows where the 'drug_ids' column matches the sorted string of chemicals
                matched_rows = database.filter(
                    database["drug_ids"] == sorted_chemicals_str
                )
                if matched_rows.is_empty():
                    st.write("No matches found.")
                    # remove search result from session state
                    if "search_result" in st.session_state:
                        del st.session_state["search_result"]

                if not matched_rows.is_empty():
                    # compute number of report_ids for each id
                    effect_counts = (
                        matched_rows.groupby("id")
                        .agg(pl.count("report_id").alias("count"))
                        .sort("count")
                    )
                    # Step 4: Join the mapping DataFrame with the report_id_counts DataFrame on the 'id' column
                    result_df = effect_counts.join(
                        pl.from_pandas(df_side_effects), on="id"
                    )
                    result_df = result_df.sort("count", descending=True).select(
                        ["condition_concept_name", "count"]
                    )
                    st.session_state["search_result"] = result_df

            # st.checkbox("Use container width", value=False, key="use_container_width")
            st.session_state.use_container_width = True

            if "search_result" in st.session_state:
                st.write("Matched Conditions:")
                st.dataframe(
                    st.session_state["search_result"],
                    use_container_width=st.session_state.use_container_width,
                )  # Convert to pandas DataFrame for Streamlit
            # Call the run_pharmageddon_predict function with the selected drugs and side effects
    else:
        show_disclaimer()


if __name__ == "__main__":
    main()
