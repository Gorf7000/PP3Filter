#!/usr/bin/env python
# coding: utf-8
# %%
# #!/usr/bin/env python
# coding: utf-8

import config
import re
import tkinter as tk
from tkinter import scrolledtext
import pandas as pd
import webbrowser

def query_oracle(text_df):
    """
    Display GUI for manual labeling of text samples.

    Parameters:
    text_df (pandas.DataFrame): DataFrame containing the text samples, index IDs, and labels.

    Returns:
    pandas.DataFrame: The modified DataFrame with the 'human_label' column filled.
    """
    text_df = initialize_dataframe(text_df)
    button_labels = load_button_labels(config.PROMPT_TEMPLATE_LABELS)
    
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    exit_loop = [False]  # Using a list to make it mutable within nested functions

    for i, row in text_df.iterrows():
        if pd.notnull(row['human_label']):
            continue

        if exit_loop[0]:  # Check if exit button was pressed
            break

        if not create_labeling_window(root, i, row, button_labels, text_df, exit_loop):
            break

    root.destroy()
    return text_df

def initialize_dataframe(text_df):
    """Ensure the DataFrame has the required columns and index."""
    if text_df.index.names == [None]:
        text_df = text_df.reset_index(drop=True)

    if 'human_label' not in text_df.columns:
        text_df['human_label'] = None

    return text_df

def load_button_labels(csv_path):
    """Load button labels from a CSV file."""
    button_labels_df = pd.read_csv(csv_path)
    return button_labels_df['label'].tolist()

def create_labeling_window(root, index, row, button_labels, text_df, exit_loop):
    """Create a new window for labeling a single text sample."""
    window = tk.Toplevel(root)
    window.title(f"Sample {index+1}/{len(text_df)}")

    # Force focus on the new window
    window.focus_force()

    article_id = row['article_ID']
    sample = row['analyze_text']

    create_article_info_frame(window, article_id)
    create_sample_text_frame(window, sample)
    create_buttons_frame(window, index, button_labels, text_df, exit_loop)

    root.wait_window(window)
    return not window.winfo_exists()

def create_article_info_frame(window, article_id):
    """Create a frame displaying article information and a clickable link."""
    info_frame = tk.Frame(window)
    info_frame.pack(pady=10)

    article_id_label = tk.Label(info_frame, text=f"Article ID: {article_id}")
    article_id_label.pack()

    link = generate_link_from_id(article_id)
    if link:
        link_label = tk.Label(info_frame, text="Click here to view the article", fg="blue", cursor="hand2")
        link_label.pack()
        link_label.bind("<Button-1>", lambda e, url=link: open_link(url))

def create_sample_text_frame(window, sample):
    """Create a frame with a scrollable text widget for the sample text."""
    text_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=80, height=35)
    text_area.pack(padx=10, pady=10)
    text_area.insert(tk.END, sample)
    text_area.configure(state='disabled')
    highlight_text(text_area, sample, 'highlight', 'explor')

def create_buttons_frame(window, index, button_labels, text_df, exit_loop):
    """Create a frame with buttons for each label and other controls."""
    global selected_label
    selected_label = None

    buttons_frame = tk.Frame(window)
    buttons_frame.pack(pady=10)

    buttons = [create_label_button(buttons_frame, idx, label) for idx, label in enumerate(button_labels)]
    flag_var = create_flag_checkbox(buttons_frame, index, text_df)
    confirm_button = create_confirm_button(buttons_frame, window, text_df, index, flag_var)
    create_exit_button(buttons_frame, window, exit_loop)

    window.bind('<KeyPress>', lambda event: on_key_press(event, buttons, confirm_button, button_labels))


def create_label_button(frame, idx, label):
    """Create a button for a single label."""
    button = tk.Button(frame, text=f"{idx+1}. {label}", command=lambda l=label: on_select(l, button))
    button.pack(side=tk.LEFT, padx=5)
    return button

def create_confirm_button(frame, window, text_df, index, flag_var):
    """Create a confirm button."""
    button = tk.Button(frame, text="Confirm", command=lambda: on_confirm(window, text_df, index, flag_var))
    button.pack(side=tk.LEFT, padx=5)
    return button


def create_exit_button(frame, window, exit_loop):
    """Create an exit button."""
    button = tk.Button(frame, text="Exit", command=lambda: on_exit(window, exit_loop))
    button.pack(side=tk.LEFT, padx=5)
    return button

def create_flag_checkbox(frame, index, text_df):
    """Create a checkbox for flagging the article."""
    flag_var = tk.IntVar(value=1 if text_df.loc[index, 'article_ID'].endswith("_FLAGGED") else 0)
    checkbox = tk.Checkbutton(frame, text="Flagged", variable=flag_var)
    checkbox.pack(side=tk.LEFT, padx=5)
    return flag_var


def generate_link_from_id(article_id):
    parts = article_id.split('_')
    if len(parts) >= 5:
        page_number = parts[2].split('p')[-1]
        date_part = parts[1]
        newspaper_id = parts[3]
        return f"https://www.loc.gov/resource/{newspaper_id}/{date_part}/ed-1/?sp={page_number}"
    return None

def open_link(url):
    webbrowser.open_new(url)

def highlight_text(text_widget, text, tag, highlight):
    text_widget.tag_remove(tag, '1.0', tk.END)
    for match in re.finditer(re.escape(highlight), text, re.IGNORECASE):
        start = f"1.0 + {match.start()} chars"
        end = f"1.0 + {match.end()} chars"
        text_widget.tag_add(tag, start, end)
    text_widget.tag_config(tag, background='yellow')

def on_key_press(event, buttons, confirm_button, labels_options):
    if event.char.isdigit():
        index = int(event.char) - 1
        if 0 <= index < len(buttons):
            buttons[index].invoke()
    elif event.keysym == 'Return':
        confirm_button.invoke()

def on_select(label, button):
    global selected_label
    selected_label = label
    for btn in button.master.winfo_children():
        if isinstance(btn, tk.Button):
            btn.config(relief=tk.RAISED, bg='SystemButtonFace')
    button.config(relief=tk.SUNKEN, bg='light blue')

def on_confirm(window, text_df, index, flag_var):
    global selected_label
    if selected_label:
        text_df.loc[index, 'human_label'] = selected_label
        # Handle the flagging
        article_id = text_df.loc[index, 'article_ID']
        if flag_var.get() == 1:
            if not article_id.endswith("_FLAGGED"):
                text_df.loc[index, 'article_ID'] = article_id + "_FLAGGED"
        else:
            if article_id.endswith("_FLAGGED"):
                text_df.loc[index, 'article_ID'] = article_id.replace("_FLAGGED", "")
        selected_label = None
    window.destroy()

def on_exit(window, exit_loop):
    exit_loop[0] = True
    global selected_label
    selected_label = None
    window.quit()
    window.destroy()

