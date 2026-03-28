#  Spam Detector — Your Personal Email Guardian

> *Because life's too short to read "Congratulations, you've won a FREE iPhone!" for the 47th time.*

---

##  What is this?

**Spam Detector** is a friendly, AI-powered command-line tool that reads any email or SMS and instantly tells you — *is this sketchy, or is it legit?*

It uses a classic but powerful combo of **TF-IDF** (a smart way to understand text) and **Naive Bayes** (a surprisingly good spam-fighting algorithm) to catch the bad guys. And it does it all with a clean, colorful terminal interface so it actually feels nice to use.

No complicated setup. No cloud subscriptions. No nonsense. Just paste a message and get an answer. 

---

##  Features

-  **Instant spam detection** — paste any message, get a verdict in seconds
-  **Confidence meter** — see exactly how sure the model is (`████████░░ 80%`)
-  **Friendly tips** — tells you what to do next ("Don't click that link!")
-  **Built-in demo mode** — run 4 sample messages to see it in action right away
-  **Remembers itself** — trains once, saves the model, loads instantly next time
-  **Session stats** — keeps count of how many spams you've caught today
-  **Graceful goodbye** — a proper farewell with your session summary

---

##  How does it actually work?

Great question! Here's the journey a message takes:

```
Your message
     ↓
Clean it up (lowercase, remove links & punctuation)
     ↓
Convert words → numbers using TF-IDF
     ↓
Naive Bayes checks the numbers
     ↓
Verdict: SPAM   or  HAM 
```

**TF-IDF** figures out which words are *really* important in your message (not just common ones like "the" or "is").

**Naive Bayes** then looks at those important words and compares them to thousands of real spam and ham messages it learned from during training.

Together, they catch spam with about **97–98% accuracy**. Pretty solid!

---

##  Getting Started

### Step 1 — Install the dependencies

```bash
pip install scikit-learn pandas rich
```

That's it. Three libraries. You're basically done.

### Step 2 — Run the program

```bash
python spam_classifier_humanised.py
```

The very first time, it will:
1. Download the SMS Spam Collection dataset (~500KB, takes a few seconds)
2. Train the model (~5 seconds)
3. Save it so you never have to wait again

After that, it loads instantly every time. 

---

##  What it looks like

```
  ✓ Found a saved model — loading it up...

  Want to see a quick demo on sample messages? [y/n]: y

  ─────────────────────────────────────────
    SPAM DETECTED

  Confidence:  ████████████████░░░░  83%
  Message:     Congratulations! You've won a FREE iPhone...
    Do NOT click any links. Mark it as spam and delete it.
  ─────────────────────────────────────────

  Your message: Hey, are we still on for lunch at 1pm?

   LOOKS GENUINE

  Confidence:  ████████████████████  97%
  Message:     Hey, are we still on for lunch at 1pm?
    This message appears safe. Always stay alert though!
  ─────────────────────────────────────────

  Check another message? [y/n]:
```

---

##  Project Structure

```
spam-detector/
│
├── spam_classifier_humanised.py   ← The main program (run this!)
├── spam_model.pkl                 ← Auto-created after first run
├── SMSSpamCollection              ← Auto-downloaded dataset
└── README.md                      ← You are here 
```

---

##  Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~97–98% |
| Spam caught correctly | Very high |
| Real emails flagged as spam | Very low |
| Training time | ~5 seconds |
| Load time (after first run) | Instant |

---

##  Try these test messages

Copy-paste any of these to see the detector in action:

**Should be flagged as SPAM **
```
Congratulations! You've won $5,000. Claim at bit.ly/win-now before midnight!
```
```
URGENT: Your account will be suspended. Verify now at secure-login.xyz
```
```
FREE entry to our prize draw! Text WIN to 80800. Ts&Cs apply.
```

**Should come back as HAM **
```
Hey! Running 10 mins late, save me a seat. See you soon!
```
```
Please find the meeting notes attached. Let me know if you have questions.
```
```
Mom, I'll be home for dinner by 7. Can you make that pasta? 
```

---

##  Want to make it even better?

Here are some fun ways to upgrade this project:

| Idea | What it does |
|------|-------------|
| Swap in Logistic Regression | Slightly better accuracy |
| Use a BERT model | State-of-the-art NLP power |
| Add a Streamlit UI | Turn it into a web app |
| Connect to Gmail API | Scan your real inbox |
| Train on your own data | Personalised spam detection |

---

##  Dependencies

| Library | What it's used for |
|---------|--------------------|
| `scikit-learn` | TF-IDF vectorizer + Naive Bayes model |
| `pandas` | Loading and handling the dataset |
| `rich` | The pretty colors and panels in the terminal |

---

##  FAQ

**Q: Does it send my messages anywhere?**
No! Everything runs 100% locally on your machine. Your messages never leave your computer.

**Q: How long does training take?**
About 5 seconds the first time. After that, it loads from the saved file instantly.

**Q: Can I use it on emails, not just SMS?**
Absolutely. Just paste the email body as your message. It works great on email text.

**Q: What if it gets something wrong?**
It's about 97–98% accurate, so it will occasionally miss a spam or flag a real message. Always use your own judgment too!

**Q: Can I retrain the model?**
Yes! Just delete `spam_model.pkl` and run the program again. It'll retrain fresh.

---

##  Credits

- **Dataset**: [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) — UCI Machine Learning Repository
- **Algorithm**: Multinomial Naive Bayes via scikit-learn
- **Terminal UI**: Built with the wonderful [Rich](https://github.com/Textualize/rich) library

---

##  License

Feel free to use, modify, and share this project. Learn from it, break it, improve it — that's the whole point! 

---

*Made with  and a healthy distrust of unsolicited prize notifications.*
