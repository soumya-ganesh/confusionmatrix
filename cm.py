import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Confusion Matrix", layout="wide")

st.title("Confusion Matrix Calculator")

st.write("Enter your true and predicted labels to visualize the confusion matrix and performance metrics.")

# --- Input Section ---
st.subheader("Enter True Labels and Predicted Labels")

true_labels = st.text_input("**True Labels** (comma-separated, e.g., 1,0,1,1,0,0)")
pred_labels = st.text_input("**Predicted Labels** (comma-separated, e.g., 1,0,0,1,0,1)")

if st.button("Generate Confusion Matrix"):
    if not true_labels or not pred_labels:
        st.warning("Please enter both true and predicted labels.")
    else:
        try:
            y_true = [int(x.strip()) for x in true_labels.split(",")]
            y_pred = [int(x.strip()) for x in pred_labels.split(",")]

            if len(y_true) != len(y_pred):
                st.error("The number of true and predicted labels must be the same.")
            else:
                # --- Force label order: 1 first, then 0 ---
                labels = [1, 0]
                cm = confusion_matrix(y_true, y_pred, labels=labels)

                # Create DataFrame with custom row/column order
                cm_df = pd.DataFrame(
                    cm,
                    index=[f"Actual {l}" for l in labels],
                    columns=[f"Predicted {l}" for l in labels]
                )

                # === Confusion Matrix Table ===
                st.subheader("Confusion Matrix (Predicted 1 before 0, Actual 1 before 0)")
                st.dataframe(
                    cm_df.style.set_properties(**{
                        'text-align': 'center',
                        'width': '60px'
                    }),
                    use_container_width=False,
                    height=120
                )

                # --- TP, FP, TN, FN (for binary classification only) ---
                if cm.shape == (2, 2):
                    # With label order [1, 0], matrix layout is:
                    # [[TP, FN],
                    #  [FP, TN]]
                    tp = cm[0, 0]
                    fn = cm[0, 1]
                    fp = cm[1, 0]
                    tn = cm[1, 1]

                    st.markdown(
                        f"""
                        <div style="font-size:18px; line-height:1.8;">
                        <b>True Positives (TP):</b> {tp} <br>
                        <b>False Positives (FP):</b> {fp} <br>
                        <b>True Negatives (TN):</b> {tn} <br>
                        <b>False Negatives (FN):</b> {fn}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.info("TP, FP, TN, FN are shown only for binary classification problems.")

                # === Evaluation Metrics ===
                st.subheader("Evaluation Metrics")

                # For direct comparison to manual calculations, use average='binary'
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
                rec = recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
                f1 = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)

                st.markdown(
                    f"""
                    <div style="font-size:18px; line-height:1.8;">
                    <b>Accuracy:</b> {acc:.2f} <br>
                    <b>Precision:</b> {prec:.2f} <br>
                    <b>Recall:</b> {rec:.2f} <br>
                    <b>F1 Score:</b> {f1:.2f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ===  Compact Heatmap ===
                st.subheader("Heatmap Visualization (1 before 0)")

                col_heatmap, _ = st.columns([1, 5])
                with col_heatmap:
                    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
                    sns.heatmap(
                        cm_df,
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        cbar=False,
                        ax=ax,
                        annot_kws={"size": 9, "ha": 'center'}
                    )
                    ax.set_xlabel("Predicted", fontsize=9)
                    ax.set_ylabel("Actual", fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=False, clear_figure=True)

        except ValueError:
            st.error(" Please enter valid numeric values separated by commas.")
