import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Confusion Matrix Visualizer", layout="wide")

st.title("Confusion Matrix Calculator")

st.write("Enter your true and predicted labels to visualize the confusion matrix and performance metrics.")

# --- Input Section ---
st.subheader("Enter True Labels and Predicted Labels")

true_labels = st.text_input("**True Labels** (comma-separated, e.g., 1,0,1,1,0,0)")
pred_labels = st.text_input("**Predicted Labels** (comma-separated, e.g., 1,0,0,1,0,1)")

if st.button("Generate Confusion Matrix"):
    if not true_labels or not pred_labels:
        st.warning("⚠️ Please enter both true and predicted labels.")
    else:
        try:
            y_true = [int(x.strip()) for x in true_labels.split(",")]
            y_pred = [int(x.strip()) for x in pred_labels.split(",")]

            if len(y_true) != len(y_pred):
                st.error("❌ The number of true and predicted labels must be the same.")
            else:
                # Compute confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                labels = sorted(list(set(y_true + y_pred)))
                cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels],
                                     columns=[f"Predicted {l}" for l in labels])

                # === 1️⃣ Confusion Matrix Table ===
                st.subheader("Confusion Matrix")
                st.dataframe(
                    cm_df.style.set_properties(**{
                        'text-align': 'center',
                        'width': '60px'
                    }),
                    use_container_width=False,
                    height=120
                )

                # === 2️⃣ Evaluation Metrics ===
                st.subheader("Evaluation Metrics")
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

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

                # === 3️⃣ Compact Heatmap ===
                st.subheader("Heatmap Visualization (Compact)")

                # Create two columns: narrow (1/6th) for heatmap, wide spacer for balance
                col_heatmap, _ = st.columns([1, 4])

                with col_heatmap:
                    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
                    sns.heatmap(
                        cm_df,
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        cbar=False,
                        ax=ax,
                        annot_kws={"size": 9, "ha": 'left'}
                    )
                    ax.set_xlabel("Predicted", fontsize=9)
                    ax.set_ylabel("Actual", fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=False, clear_figure=True)

        except ValueError:
            st.error("⚠️ Please enter valid numeric values separated by commas.")

