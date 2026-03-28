import re
import os
import sys
import time
import pickle
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
    from rich import box
except ImportError:
    print("'rich' library not found. Install it with: pip install rich")
    sys.exit(1)

console = Console()

MODEL_FILE = "spam_model.pkl"
DATA_FILE  = "SMSSpamCollection"




def welcome():
    console.clear()
    console.print()
    console.print(Panel.fit(
        "[bold cyan]SPAM DETECTOR[/bold cyan]  🛡️\n"
        "[dim]Your personal AI-powered email guardian[/dim]",
        border_style="cyan",
        padding=(1, 6),
    ))
    console.print()
    console.print("  [dim]Paste any email or SMS and I'll tell you[/dim]")
    console.print("  [dim]whether it's spam or a genuine message.[/dim]")
    console.print()




def download_dataset():
    if os.path.exists(DATA_FILE):
        console.print("  [green]✓[/green] Dataset is already available.\n")
        return

    console.print("  [yellow]↓[/yellow] Downloading dataset for the first time...\n")

    
    sources = [
        {
            "label": "GitHub mirror (raw TSV)",
            "url":   "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset/master/spam.csv",
            "type":  "csv",
        },
        {
            "label": "GitHub mirror 2 (raw TSV)",
            "url":   "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
            "type":  "tsv",
        },
        {
            "label": "UCI repository (zip)",
            "url":   "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
            "type":  "zip",
        },
    ]

    for source in sources:
        try:
            console.print(f"  Trying: [dim]{source['label']}[/dim]")
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                          transient=True) as progress:
                task = progress.add_task("  Fetching...", total=None)

                if source["type"] == "zip":
                    zip_path = "smsspamcollection.zip"
                    urllib.request.urlretrieve(source["url"], zip_path)
                    with zipfile.ZipFile(zip_path, "r") as z:
                        z.extractall(".")
                    os.remove(zip_path)

                elif source["type"] == "tsv":
                    tmp = "sms_tmp.tsv"
                    urllib.request.urlretrieve(source["url"], tmp)
                    df = pd.read_csv(tmp, sep="\t", header=None, names=["label", "text"])
                    df.to_csv(DATA_FILE, sep="\t", index=False, header=False)
                    os.remove(tmp)

                elif source["type"] == "csv":
                    tmp = "sms_tmp.csv"
                    urllib.request.urlretrieve(source["url"], tmp)
                    df = pd.read_csv(tmp, encoding="latin-1", usecols=[0, 1])
                    df.columns = ["label", "text"]
                    df["label"] = df["label"].str.lower().str.strip()
                    df.to_csv(DATA_FILE, sep="\t", index=False, header=False)
                    os.remove(tmp)

                progress.advance(task)

            console.print("  [green]✓[/green] Dataset ready!\n")
            return  

        except Exception as e:
            console.print(f"  [yellow]⚠[/yellow]  That didn't work ({type(e).__name__}). Trying next source...\n")

    
    console.print(Panel(
        "  [bold red]Could not download the dataset.[/bold red]\n\n"
        "  Please download it manually:\n"
        "  [cyan]https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset[/cyan]\n\n"
        "  Save the file as [bold]SMSSpamCollection[/bold] (tab-separated, no header)\n"
        "  in the same folder as this script, then run again.",
        border_style="red", padding=(0, 2)
    ))
    sys.exit(1)


def load_data():
    return pd.read_csv(DATA_FILE, sep="\t", header=None,
                       names=["label", "text"], encoding="latin-1")




def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text




def train_model():
    console.print("\n  [bold]Training the model — just a moment...[/bold]\n")

    df = load_data()
    df["label_enc"]  = df["label"].map({"ham": 0, "spam": 1})
    df["clean_text"] = df["text"].apply(clean_text)

    X = df["clean_text"]
    y = df["label_enc"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  transient=True) as progress:

        t1 = progress.add_task("  Converting text to numbers (TF-IDF)...", total=None)
        vectorizer  = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec  = vectorizer.transform(X_test)
        progress.advance(t1)

        t2 = progress.add_task("  Teaching Naive Bayes to spot spam...", total=None)
        model = MultinomialNB(alpha=0.1)
        model.fit(X_train_vec, y_train)
        progress.advance(t2)

    y_pred   = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred) * 100
    cm       = confusion_matrix(y_test, y_pred)

    table = Table(box=box.ROUNDED, border_style="cyan", header_style="bold cyan")
    table.add_column("Metric",           style="dim", width=24)
    table.add_column("Value",            justify="right")

    table.add_row("Total messages",      f"{len(df):,}")
    table.add_row("Spam messages",       f"[red]{df['label_enc'].sum():,}[/red]")
    table.add_row("Ham messages",        f"[green]{(df['label_enc']==0).sum():,}[/green]")
    table.add_row("─" * 22,             "─" * 8)
    table.add_row("Model accuracy",     f"[bold green]{accuracy:.1f}%[/bold green]")
    table.add_row("Spam caught (TP)",   f"[green]{cm[1][1]}[/green]")
    table.add_row("Spam missed (FN)",   f"[yellow]{cm[1][0]}[/yellow]")
    table.add_row("False alarms (FP)",  f"[yellow]{cm[0][1]}[/yellow]")

    console.print(Panel(table, title="[bold]Training Complete 🎉[/bold]",
                        border_style="green", padding=(0, 1)))

    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f)

    console.print(f"\n  [green]✓[/green] Model saved as [cyan]{MODEL_FILE}[/cyan] "
                  f"— no retraining needed next time!\n")

    return model, vectorizer


def load_model():
    with open(MODEL_FILE, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["vectorizer"]




def predict_and_show(text: str, model, vectorizer):
    cleaned    = clean_text(text)
    vec        = vectorizer.transform([cleaned])
    pred       = model.predict(vec)[0]
    prob       = model.predict_proba(vec)[0]
    confidence = max(prob) * 100
    is_spam    = pred == 1

    
    console.print("\n  [dim]Analysing", end="")
    for _ in range(4):
        time.sleep(0.2)
        console.print("[dim].[/dim]", end="", highlight=False)
    console.print()

    
    filled    = int(confidence / 5)
    empty     = 20 - filled
    bar_color = "red" if is_spam else "green"
    bar       = f"[{bar_color}]{'█' * filled}[/{bar_color}][dim]{'░' * empty}[/dim]"

    if is_spam:
        headline   = "[bold red]  🚨  SPAM DETECTED[/bold red]"
        tip        = "Do NOT click any links. Mark it as spam and delete it."
        border_col = "red"
    else:
        headline   = "[bold green]  ✅  LOOKS GENUINE[/bold green]"
        tip        = "This message appears safe. Always stay alert though!"
        border_col = "green"

    preview = text[:75] + ("..." if len(text) > 75 else "")

    content = (
        f"\n{headline}\n\n"
        f"  [dim]Confidence:[/dim]  {bar}  [bold]{confidence:.0f}%[/bold]\n\n"
        f"  [dim]Message:[/dim]     [italic]{preview}[/italic]\n\n"
        f"  💡  [dim]{tip}[/dim]\n"
    )

    console.print(Panel(content, border_style=border_col, padding=(0, 1)))




SAMPLES = [
    ("Spam — Prize scam",    "Congratulations! You've won a FREE iPhone. Click here to claim NOW: bit.ly/free-prize"),
    ("Spam — Bank phishing", "URGENT: Your bank account has been suspended. Verify at secure-login.xyz immediately."),
    ("Ham  — Friendly text", "Hey! Are we still on for lunch tomorrow at 1pm? Let me know!"),
    ("Ham  — Work email",    "Hi, just a reminder that the project report is due by end of day Friday. Thanks!"),
]

def run_samples(model, vectorizer):
    console.print("\n  [bold]Running a quick demo on sample messages...[/bold]\n")
    for label, msg in SAMPLES:
        console.print(f"  [dim]Sample ({label}):[/dim]")
        predict_and_show(msg, model, vectorizer)
        time.sleep(0.3)




def main():
    welcome()

    
    if os.path.exists(MODEL_FILE):
        console.print("  [green]✓[/green] Found a saved model — loading it up...\n")
        model, vectorizer = load_model()
    else:
        console.print("  [yellow]![/yellow] No saved model found. Let me train one first.\n")
        download_dataset()
        model, vectorizer = train_model()

    
    if Confirm.ask("  Want to see a quick demo on sample messages?", default=True):
        run_samples(model, vectorizer)

   
    console.print()
    console.print(Panel(
        "  [bold]All set![/bold] Paste any message below.\n"
        "  [dim]Type [cyan]quit[/cyan] anytime to exit.[/dim]",
        border_style="dim", padding=(0, 1)
    ))

    session_total = 0
    session_spam  = 0

    while True:
        console.print()
        try:
            user_input = Prompt.ask("  [bold cyan]Your message[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            console.print("  [dim]Nothing to check — paste a message first![/dim]")
            continue

        if user_input.lower() in ("quit", "exit", "q", "bye"):
            break

        predict_and_show(user_input, model, vectorizer)
        session_total += 1

        pred = model.predict(vectorizer.transform([clean_text(user_input)]))[0]
        if pred == 1:
            session_spam += 1

        
        if session_total % 3 == 0:
            console.print(
                f"\n  [dim]Session so far: [bold]{session_total}[/bold] checked, "
                f"[bold red]{session_spam}[/bold red] spam caught.[/dim]"
            )

        if not Confirm.ask("\n  [dim]Check another message?[/dim]", default=True):
            break

   
    console.print()
    console.print(Panel(
        f"  [bold]Thanks for using Spam Detector![/bold] 👋\n\n"
        f"  You checked [cyan]{session_total}[/cyan] message(s) today and caught "
        f"[red]{session_spam}[/red] spam.\n"
        f"  [dim]Stay safe out there![/dim]",
        border_style="cyan", padding=(0, 2)
    ))
    console.print()


if __name__ == "__main__":
    main()
