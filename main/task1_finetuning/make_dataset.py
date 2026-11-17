"""Create deterministic instruction/target examples and disjoint train/val/test splits."""
from pathlib import Path
import json, random, yaml

ROOT = Path(__file__).parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
OUT = ROOT / CFG["dataset_dir"]
OUT.mkdir(exist_ok=True)

ACTIONS = {
    "create": ["create", "add", "open", "make", "set up", "log"],
    "update": ["update", "change", "edit", "move", "revise", "modify"],
    "delete": ["delete", "remove", "close permanently", "erase"],
    "search": ["find", "search for", "look up", "show me", "locate"],
}
OBJECTS = {
    "task": ["task", "todo", "work item", "action item"],
    "meeting": ["meeting", "calendar event", "sync", "appointment"],
    "document": ["document", "doc", "file", "specification"],
    "ticket": ["ticket", "issue", "bug report", "support case"],
}
PRIORITIES = {
    "low": ["low priority", "when you get a chance", "not urgent", "low"],
    "medium": ["normal priority", "standard priority", "medium priority", "medium"],
    "high": ["urgent", "high priority", "as soon as possible", "critical"],
}
ASSIGNEES = ["Ava", "Noah", "Maya", "Ishan", "Priya", "the platform team", "the QA team", None]
DEADLINES = [-1, 0, 1, 2, 3, 5, 7, 14, 30]
DEADLINE_TEXT = {
    -1: ["with no deadline", "there is no due date", "leave the deadline blank"],
    0: ["today", "by end of day", "before the day ends"],
    1: ["tomorrow", "by this time tomorrow", "in one day"],
    2: ["in two days", "two days from now"],
    3: ["in three days", "by the end of the week (three days)"],
    5: ["in five days", "within five days"],
    7: ["next week", "in seven days"],
    14: ["in two weeks", "within fourteen days"],
    30: ["in a month", "within thirty days"],
}

def make_one(rng, idx):
    action = rng.choice(list(ACTIONS))
    obj = rng.choice(list(OBJECTS))
    priority = rng.choice(list(PRIORITIES))
    assignee = rng.choice(ASSIGNEES)
    days = rng.choice(DEADLINES)
    verb = rng.choice(ACTIONS[action])
    noun = rng.choice(OBJECTS[obj])
    p = rng.choice(PRIORITIES[priority])
    d = rng.choice(DEADLINE_TEXT[days])
    assignee_phrase = "" if assignee is None else f" and assign it to {assignee}"
    templates = [
        f"Please {verb} a {noun}, mark it {p}, {d}{assignee_phrase}.",
        f"I need you to {verb} the {noun}. Set priority to {p}; deadline: {d}.{assignee_phrase}",
        f"{verb.capitalize()} {noun} {d}, priority {p}{assignee_phrase}.",
        f"Could you {verb} a {noun} for me? It is {p}, due {d}{assignee_phrase}.",
    ]
    prompt = rng.choice(templates)
    target = {
        "action": action, "object": obj, "priority": priority,
        "due_in_days": days, "assignee": assignee
    }
    return {"id": f"ex_{idx:04d}", "instruction": prompt,
            "input": "", "output": json.dumps(target, ensure_ascii=False)}

def main():
    rng = random.Random(CFG["seed"])
    total = sum(CFG[k] for k in ("train_examples", "validation_examples", "test_examples"))
    rows = [make_one(rng, i) for i in range(total)]
    rng.shuffle(rows)
    ntr, nv = CFG["train_examples"], CFG["validation_examples"]
    splits = {"train": rows[:ntr], "validation": rows[ntr:ntr+nv], "test": rows[ntr+nv:]}
    for split, data in splits.items():
        path = OUT / f"{split}.jsonl"
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in data))
        print(f"{split}: {len(data)} -> {path}")
    print("Split sizes are disjoint. Keep test.jsonl out of training.")

if __name__ == "__main__":
    main()
