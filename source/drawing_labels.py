import pandas as pd
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("/home/yosakoi/Work/momo_v2/data/BTCUSDT/label_15.csv")
df_label_close = df[["top", "bot", "close"]]
# Filter data for a specific time period (e.g., one year)
# Adjust the ranxge to match your dataset's index or datetime column


# ----------------- First Image: Original and Annotated Data -----------------
fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # Two plots side by side

# Plot 1: Original data
axes[0].plot(df_label_close.index, df_label_close["close"], label="Close", color="black", lw=1)
axes[0].set_title("Original Data", fontsize=16)
axes[0].set_xlabel("Index", fontsize=12)
axes[0].set_ylabel("Close", fontsize=12)
axes[0].grid(alpha=0.5)
axes[0].legend(fontsize=12)
axes[0].tick_params(axis="both", which="major", labelsize=10)

# Plot 2: Annotated data with 'top' and 'bot'
axes[1].plot(df_label_close.index, df_label_close["close"], label="Close", color="black", lw=1)
axes[1].scatter(
    df_label_close.index[df_label_close["top"]],
    df_label_close["close"][df_label_close["top"]],
    color="red",
    label="Top True",
    zorder=5,
    alpha=1,
    s=1,
)
axes[1].scatter(
    df_label_close.index[df_label_close["bot"]],
    df_label_close["close"][df_label_close["bot"]],
    color="blue",
    label="Bot True",
    zorder=5,
    alpha=1,
    s=1,
)
axes[1].set_title("Annotated Data with Top and Bot Markers", fontsize=16)
axes[1].set_xlabel("Index", fontsize=12)
axes[1].set_ylabel("Close", fontsize=12)
axes[1].grid(alpha=0.5)
axes[1].legend(fontsize=12)
axes[1].tick_params(axis="both", which="major", labelsize=10)

# Adjust layout and save the first image
plt.tight_layout()
plt.savefig("/home/yosakoi/Work/momo_v2/data/BTCUSDT/original_and_annotated.png")
plt.close()  # Close the first figure

# ----------------- Second Image: Zoomed-in Data -----------------
start_index = 115000  # Replace with the actual start index for the year
end_index = 120000    # Replace with the actual end index for the year
df_zoomed = df_label_close.iloc[start_index:end_index]
fig, ax = plt.subplots(figsize=(12, 8))  # Single plot for zoomed-in data

# Plot: Zoomed-in data for a specific time period
ax.plot(df_zoomed.index, df_zoomed["close"], label="Close", color="black", lw=1)
ax.scatter(
    df_zoomed.index[df_zoomed["top"]],
    df_zoomed["close"][df_zoomed["top"]],
    color="red",
    label="Top True",
    zorder=5,
    alpha=1,
    s=1,  # Increase size for better visibility
)
ax.scatter(
    df_zoomed.index[df_zoomed["bot"]],
    df_zoomed["close"][df_zoomed["bot"]],
    color="blue",
    label="Bot True",
    zorder=5,
    alpha=1,
    s=1,  # Increase size for better visibility
)
ax.set_title("Zoomed-in Data with Top and Bot Markers", fontsize=16)
ax.set_xlabel("Index", fontsize=12)
ax.set_ylabel("Close", fontsize=12)
ax.grid(alpha=0.5)
ax.legend(fontsize=12)
ax.tick_params(axis="both", which="major", labelsize=10)

# Save the second image
plt.tight_layout()
plt.savefig("/home/yosakoi/Work/momo_v2/data/BTCUSDT/zoomed_in_data.png")
plt.close()  # Close the second figure

# ----------------- Third Image: Zoomed-in Data -----------------
start_index = 105000  # Replace with the actual start index for the year
end_index = 106000    # Replace with the actual end index for the year
df_zoomed = df_label_close.iloc[start_index:end_index]
fig, ax = plt.subplots(figsize=(12, 8))  # Single plot for zoomed-in data

# Plot: Zoomed-in data for a specific time period
ax.plot(df_zoomed.index, df_zoomed["close"], label="Close", color="black", lw=1)
ax.scatter(
    df_zoomed.index[df_zoomed["top"]],
    df_zoomed["close"][df_zoomed["top"]],
    color="red",
    label="Top True",
    zorder=5,
    alpha=1,
    s=1,  # Increase size for better visibility
)
ax.scatter(
    df_zoomed.index[df_zoomed["bot"]],
    df_zoomed["close"][df_zoomed["bot"]],
    color="blue",
    label="Bot True",
    zorder=5,
    alpha=1,
    s=1,  # Increase size for better visibility
)
ax.set_title("Zoomed-in Data with Top and Bot Markers", fontsize=16)
ax.set_xlabel("Index", fontsize=12)
ax.set_ylabel("Close", fontsize=12)
ax.grid(alpha=0.5)
ax.legend(fontsize=12)
ax.tick_params(axis="both", which="major", labelsize=10)

# Save the second image
plt.tight_layout()
plt.savefig("/home/yosakoi/Work/momo_v2/data/BTCUSDT/zoomed_in_data_v2.png")
plt.close()  # Close the second figure