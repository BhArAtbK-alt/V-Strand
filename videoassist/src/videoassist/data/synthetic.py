"""Staged-video stand-ins: evidence records (what extractors would output) + human-style ground truth."""
import random
OPS = {
 "router_installation": [
  ("unbox the router and check the package contents", "technician opening a cardboard box and lifting out a new router", ["Checking everything is in the box"], None),
  ("mount the router on the wall bracket", "technician fixing the router onto a wall mounted bracket with screws", ["Mounting it up on the bracket"], None),
  ("connect the power cable to the router", "technician plugging a black power adapter cable into the router power port", ["Plugging in the power now"], None),
  ("connect the ethernet cable from the ONT to the WAN port", "technician inserting a yellow ethernet cable into the WAN port of the router", ["WAN cable goes from the ONT into the yellow port"], None),
  ("wait for the status LED to turn green", "close view of router front panel LEDs with the power LED turning green", ["Now we wait for the green light"], "PWR OK  WAN LINK"),
  ("log in to the admin page and set the Wi-Fi password", "laptop screen showing the router admin login page and settings form", ["Setting the Wi-Fi password"], "Router Admin Login"),
  ("run a speed test to verify the connection", "laptop screen showing a speed test gauge with the result", [], "Download {mbps} Mbps")],
 "router_troubleshooting": [
  ("observe that the internet LED is blinking red", "close view of router front panel with the internet LED blinking red", ["Internet light is blinking red"], None),
  ("read the error code on the router display", "router status display showing an error message with a code", ["There is an error code on the screen"], "ERROR {code} WAN TIMEOUT"),
  ("check the ethernet cable at the ONT", "technician inspecting an ethernet cable plugged into the ONT box", ["Checking the cable at the ONT end"], None),
  ("replace the faulty ethernet cable", "technician swapping a damaged ethernet cable with a new blue cable", [], None),
  ("reboot the router by unplugging the power cable", "technician pulling the power cable from the router then plugging it back", ["Let's power cycle it"], None),
  ("confirm the internet LED turns green", "router front panel with the internet LED steady green", ["Internet light is green now"], "WAN LINK UP"),
  ("ping the gateway from the laptop", "laptop terminal window showing ping replies", [], "Reply from 192.168.1.1 time=2ms")],
 "device_replacement": [
  ("power off the old router", "technician switching off the old router and unplugging the power adapter", ["Powering this one off"], None),
  ("label and disconnect the cables from the old router", "technician attaching paper labels to cables and unplugging them from the old router", ["Labeling every cable first"], None),
  ("remove the old router from the wall bracket", "technician lifting the old router off the wall bracket", [], None),
  ("mount the new router on the wall bracket", "technician placing a new router onto the wall bracket", ["New unit goes on the bracket"], "Model {model}"),
  ("reconnect the ethernet and power cables to the new router", "technician plugging labeled ethernet and power cables into the new router", ["Reconnecting following the labels"], None),
  ("restore the saved configuration backup", "laptop screen showing a configuration restore progress bar", ["Restoring the backup config"], "Restore Complete"),
  ("verify internet connectivity on the new router", "laptop browser loading a website with the router LEDs green in view", [], "Ping OK")]}
UNANSWERABLE = ["Did the technician configure a VLAN?", "When does the technician replace the SFP module?",
                "Did the technician take a photo of the serial number?", "When does the technician call the customer?"]

def generate(n=60, seed=7):
    rnd, vids = random.Random(seed), []
    ops = list(OPS)
    for v in range(n):
        op, t, ev, events = ops[v % 3], 0.0, [], []
        fill = {"mbps": rnd.randint(40, 940), "code": f"E-{rnd.randint(1000, 4999)}", "model": f"RX-{rnd.randint(2000, 6000)}"}
        vid, key = f"{op[:3]}_{v:03d}", None
        for label, vis, speech, ocr in OPS[op]:
            s, e = t, t + rnd.randint(8, 16)
            for a in range(int(s), int(e), 2):
                ev.append(dict(modality="visual", t0=a, t1=min(a + 2, e), text=vis))
            if speech and rnd.random() < 0.65:
                ev.append(dict(modality="speech", t0=s + 1, t1=min(s + 4, e), text=speech[0]))
            if ocr:
                txt = ocr.format(**fill)
                ev.append(dict(modality="ocr", t0=s + 2, t1=min(s + 6, e), text=txt))
            events.append(dict(label=label, start=s, end=e, key=fill["code"] if "{code}" in (ocr or "") else None))
            t = e + rnd.randint(0, 2)
        qs, idx = [], list(range(len(events)))
        for i in rnd.sample(idx, 3):
            qs.append(dict(q=f"When does the technician {events[i]['label']}?", type="when", i=i))
        for i in rnd.sample(idx[1:], 2):
            qs.append(dict(q=f"What does the technician do after they {events[i-1]['label']}?", type="after", i=i))
        j = rnd.choice(idx[:-1]); qs.append(dict(q=f"What does the technician do before they {events[j+1]['label']}?", type="before", i=j))
        ci = next((i for i, x in enumerate(events) if x["key"]), None)
        qs.append(dict(q="What error code is shown on the router?", type="code", i=ci) if ci is not None
                  else dict(q="What error code is shown on the router?", type="unanswerable", i=None))
        for q in rnd.sample(UNANSWERABLE, 2): qs.append(dict(q=q, type="unanswerable", i=None))
        for q in qs:
            ev_i = events[q["i"]] if q["i"] is not None else None
            q.update(answerable=ev_i is not None, start=ev_i and ev_i["start"], end=ev_i and ev_i["end"], key=ev_i and ev_i["key"])
        vids.append(dict(video_id=vid, operation=op, evidence=[dict(video_id=vid, **x) for x in ev],
                         gt=dict(events=events, queries=qs)))
    return vids
