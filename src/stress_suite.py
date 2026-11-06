"""Make the schema-stress suite in data/stress_suite/, and check it.

Usage (from the repository root, in the BFCL environment):
    python src/stress_suite.py            # write the question and answer files
    python src/stress_suite.py --check    # check the files on disk (see check() below)

Seven groups of 25 questions. Six each stress one feature of a function's schema; the seventh (plain) is a
control without any of them:

    stress_plain     a flat function with two or three required strings and integers, ASCII only (control)
    stress_nested    objects as arguments: one object, a list of objects, objects nested 2 and 3 levels deep
    stress_enum      a code from a fixed list (4, 12 or 30 codes); the question names the meaning, and the
                     description says which code means what
    stress_number    integer vs. float: thousands separators, numbers in words, negatives, decimals, both kinds
    stress_optional  2, 4 or 8 optional arguments with defaults; the question sets none, one or two of them
    stress_unicode   strings with accents, non-Latin scripts and emoji, to be copied exactly
    stress_menu      50 functions to choose from (5 actions x 10 objects, so many names are near duplicates)

Every question has exactly one correct call, and every value in it is stated in the question (enum codes: in
the function description). The files follow BFCL's layout, one JSON object per line: questions in
data/stress_suite/<group>.json, answers in data/stress_suite/possible_answer/<group>.json. Answers follow BFCL's
convention: a list of accepted values per argument, where "" means the argument may be left out. Each question
also has a "level" (e.g. the nesting depth), so results can be split further.

Menu functions have no argument descriptions: with them, 50 functions do not fit the 4,096-token prompt budget
(max_model_len 8192 minus 4,096 answer tokens). Everything is drawn with fixed seeds, so the files can be made
again exactly.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "stress_suite"
SEED = 42
PER_GROUP = 25
GROUPS = ["stress_plain", "stress_nested", "stress_enum", "stress_number", "stress_optional", "stress_unicode",
          "stress_menu"]

# ---------------------------------------------------------------- pools (invented names, no real people)
PEOPLE = ["Ana Silva", "Ben Carter", "Chloe Martin", "David Kim", "Elena Rossi", "Farid Khan", "Grace Lee",
          "Hugo Bernard", "Irene Novak", "Jonas Berg", "Kavya Rao", "Liam Walsh", "Maya Cohen", "Noah Fischer",
          "Olivia Brown", "Pedro Alves", "Quinn Taylor", "Rosa Diaz", "Sam Okafor", "Tara Singh"]
CITIES = ["Lisbon", "Denver", "Oslo", "Nairobi", "Toronto", "Dublin", "Austin", "Hamburg", "Perth", "Seoul",
          "Lima", "Porto", "Cairo", "Boston", "Madrid", "Vienna", "Quito", "Riga", "Accra", "Hanoi", "Bergen",
          "Tucson", "Leeds", "Kyoto", "Malmo", "Durban", "Tampa", "Lyon", "Turin", "Graz"]
STREETS = ["Main Street", "Harbor Road", "Elm Avenue", "Park Lane", "Mill Road", "Station Street", "Oak Drive",
           "River Walk", "Market Square", "Hill Road"]
RESTAURANTS = ["Blue Fig", "The Copper Pot", "Luna Rossa", "Green Garden", "Harbor House", "Saffron Lane",
               "Little Tokyo", "Olive Tree", "The Red Door", "Maple Grill"]
BOOKS = [("The Silent Coast", "Mara Quinn"), ("Glass Rivers", "Tom Ashby"), ("A Map of Small Things", "Iris Vale"),
         ("Northern Lights Fall", "Owen Hart"), ("The Last Orchard", "June Patel"), ("Salt and Iron", "Leo Marsh"),
         ("Paper Moons", "Nina Ford"), ("The Clockmaker's Son", "Eli Brandt")]
TASKS = ["Renew passport", "Call the plumber", "Submit tax form", "Book dentist visit", "Water the plants",
         "Pay rent", "Update resume", "Order printer ink", "Fix the bike", "Plan team lunch"]
SUBJECTS = ["Quarterly report", "Project kickoff", "Invoice reminder", "Team offsite", "Contract renewal",
            "Weekly update", "Budget review", "Launch plan"]
EVENTS = ["Spring Gala", "Tech Meetup", "Book Club Night", "Charity Run", "Jazz Evening", "Design Sprint",
          "Science Fair", "Film Night"]
VENUES = ["City Hall", "Riverside Center", "The Grand Hall", "Oak Pavilion", "Union Theater", "Hilltop Club"]
MODELS = ["X200", "Pro 14", "Air 3", "Max 7", "Nova S", "Edge 5"]


def email(name):
    return name.lower().replace(" ", ".") + "@example.com"


def code(rng, prefix, digits=4):
    return f"{prefix}-{rng.randint(10 ** (digits - 1), 10 ** digits - 1)}"


def date(rng):
    return f"2026-{rng.randint(11, 12):02d}-{rng.randint(1, 28):02d}"


ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def words(n):
    """English words for 0-99, e.g. 42 -> forty-two."""
    if n < 20:
        return ONES[n]
    return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])


class Deck:
    """Draws from a list without repeats until it runs out, then starts a new shuffled round."""

    def __init__(self, rng, items):
        self.rng, self.items, self.left = rng, list(items), []

    def draw(self):
        if not self.left:
            self.left = self.rng.sample(self.items, len(self.items))
        return self.left.pop()


# ---------------------------------------------------------------- building blocks
def prop(type_, description, **extra):
    return {"type": type_, "description": description, **extra}


def function(name, description, properties, required):
    return {"name": name, "description": description,
            "parameters": {"type": "dict", "properties": properties, "required": required}}


def case(level, text, functions, answer, reference):
    """answer: {function name: {argument: [accepted values]}}; reference: one correct call, for the checks."""
    return {"level": level, "text": text, "functions": functions, "answer": answer, "reference": reference}


def simple(level, text, fn, accepted):
    """A case with one function whose answer accepts exactly one value per argument."""
    answer = {fn["name"]: {k: [v] for k, v in accepted.items()}}
    return case(level, text, [fn], answer, {"name": fn["name"], "arguments": dict(accepted)})


def levels(sizes):
    """[(level, count), ...] -> a list of 25 level names."""
    out = [level for level, n in sizes for _ in range(n)]
    assert len(out) == PER_GROUP, out
    return out


# ---------------------------------------------------------------- the seven groups
def plain(rng):
    cases = []
    restaurants, cities, tasks, books, people, subjects = (Deck(rng, x) for x in (RESTAURANTS, CITIES, TASKS, BOOKS,
                                                                                   PEOPLE, SUBJECTS))
    for i in range(PER_GROUP):
        t = i % 5
        if t == 0:
            r, n, h = restaurants.draw(), rng.randint(2, 9), f"{rng.randint(17, 21)}:{rng.choice(['00', '30'])}"
            fn = function("book_table", "Book a table at a restaurant.",
                          {"restaurant": prop("string", "Name of the restaurant."),
                           "party_size": prop("integer", "Number of people."),
                           "time": prop("string", "Time of the booking, HH:MM (24-hour clock).")},
                          ["restaurant", "party_size", "time"])
            cases.append(simple("control", f"Book a table at {r} for {n} people at {h}.", fn,
                                {"restaurant": r, "party_size": n, "time": h}))
        elif t == 1:
            c, d = cities.draw(), rng.randint(2, 10)
            fn = function("get_weather_forecast", "Get the weather forecast for a city.",
                          {"city": prop("string", "Name of the city."),
                           "days": prop("integer", "Number of days to forecast.")}, ["city", "days"])
            cases.append(simple("control", f"What will the weather be like in {c} over the next {d} days?", fn,
                                {"city": c, "days": d}))
        elif t == 2:
            title, due = tasks.draw(), date(rng)
            fn = function("add_task", "Add a task to the to-do list.",
                          {"title": prop("string", "Title of the task."),
                           "due_date": prop("string", "Due date, YYYY-MM-DD.")}, ["title", "due_date"])
            cases.append(simple("control", f'Add a task called "{title}" due on {due}.', fn,
                                {"title": title, "due_date": due}))
        elif t == 3:
            title, author = books.draw()
            fn = function("find_book", "Find a book in the library catalog.",
                          {"title": prop("string", "Title of the book."),
                           "author": prop("string", "Name of the author.")}, ["title", "author"])
            cases.append(simple("control", f'Find the book "{title}" by {author} in the library.', fn,
                                {"title": title, "author": author}))
        else:
            to, subject = email(people.draw()), subjects.draw()
            fn = function("send_email", "Send an email.",
                          {"to": prop("string", "Email address of the recipient."),
                           "subject": prop("string", "Subject line.")}, ["to", "subject"])
            cases.append(simple("control", f'Send an email to {to} with the subject "{subject}".', fn,
                                {"to": to, "subject": subject}))
    return cases


def nested(rng):
    cases = []
    for i, level in enumerate(levels([("object", 7), ("list_of_objects", 6), ("depth_2", 6), ("depth_3", 6)])):
        alt = i % 2
        if level == "object" and alt == 0:
            name, city = rng.choice(PEOPLE), rng.choice(CITIES)
            contact = {"name": name, "email": email(name), "city": city}
            fn = function("create_contact", "Save a new contact.",
                          {"contact": prop("dict", "The contact to save.", properties={
                              "name": prop("string", "Full name."), "email": prop("string", "Email address."),
                              "city": prop("string", "City where the contact lives.")},
                              required=["name", "email", "city"])}, ["contact"])
            text = f"Save a new contact: {name}, email {contact['email']}, who lives in {city}."
            answer = {"create_contact": {"contact": [{k: [v] for k, v in contact.items()}]}}
            cases.append(case(level, text, [fn], answer, {"name": fn["name"], "arguments": {"contact": contact}}))
        elif level == "object":
            device = {"serial_number": code(rng, "SN", 6), "model": rng.choice(MODELS), "owner": rng.choice(PEOPLE)}
            fn = function("register_device", "Register a device.",
                          {"device": prop("dict", "The device to register.", properties={
                              "serial_number": prop("string", "Serial number."),
                              "model": prop("string", "Model name."), "owner": prop("string", "Name of the owner.")},
                              required=["serial_number", "model", "owner"])}, ["device"])
            text = (f"Register the device with serial number {device['serial_number']}, model {device['model']}, "
                    f"owned by {device['owner']}.")
            answer = {"register_device": {"device": [{k: [v] for k, v in device.items()}]}}
            cases.append(case(level, text, [fn], answer, {"name": fn["name"], "arguments": {"device": device}}))
        elif level == "list_of_objects" and alt == 1:
            order_id = code(rng, "ORD")
            items = [{"sku": code(rng, "SKU", 3), "quantity": rng.randint(1, 9)} for _ in range(rng.randint(2, 3))]
            fn = function("add_order_items", "Add items to an order.",
                          {"order_id": prop("string", "ID of the order."),
                           "items": prop("array", "Items to add, in the order given.", items={
                               "type": "dict", "properties": {"sku": prop("string", "Product code."),
                                                              "quantity": prop("integer", "Number of units.")},
                               "required": ["sku", "quantity"]})}, ["order_id", "items"])
            listed = " and ".join(f"{it['quantity']} x {it['sku']}" for it in items)
            text = f"Add these items to order {order_id}, in this order: {listed}."
            answer = {"add_order_items": {"order_id": [order_id],
                                          "items": [[{k: [v] for k, v in it.items()} for it in items]]}}
            cases.append(case(level, text, [fn], answer,
                              {"name": fn["name"], "arguments": {"order_id": order_id, "items": items}}))
        elif level == "list_of_objects":
            stops = [{"city": c, "nights": rng.randint(1, 5)} for c in rng.sample(CITIES, rng.randint(2, 3))]
            fn = function("plan_trip", "Plan a trip with several stops.",
                          {"stops": prop("array", "The stops, in travel order.", items={
                              "type": "dict", "properties": {"city": prop("string", "City of the stop."),
                                                             "nights": prop("integer", "Nights to stay.")},
                              "required": ["city", "nights"]})}, ["stops"])
            listed = ", then ".join(f"{s['nights']} night{'s' if s['nights'] > 1 else ''} in {s['city']}"
                                    for s in stops)
            text = f"Plan a trip with these stops in order: {listed}."
            answer = {"plan_trip": {"stops": [[{k: [v] for k, v in s.items()} for s in stops]]}}
            cases.append(case(level, text, [fn], answer, {"name": fn["name"], "arguments": {"stops": stops}}))
        elif level == "depth_2" and alt == 0:
            name = rng.choice(PEOPLE)
            order = {"order_id": code(rng, "ORD"), "customer": {"name": name, "email": email(name)}}
            fn = function("create_order", "Create an order.",
                          {"order": prop("dict", "The order.", properties={
                              "order_id": prop("string", "ID of the order."),
                              "customer": prop("dict", "The customer.", properties={
                                  "name": prop("string", "Full name."), "email": prop("string", "Email address.")},
                                  required=["name", "email"])}, required=["order_id", "customer"])}, ["order"])
            text = f"Create order {order['order_id']} for the customer {name}, email {order['customer']['email']}."
            answer = {"create_order": {"order": [{"order_id": [order["order_id"]], "customer": [order["customer"]]}]}}
            cases.append(case(level, text, [fn], answer, {"name": fn["name"], "arguments": {"order": order}}))
        elif level == "depth_2":
            parcel = {"tracking_code": code(rng, "TRK", 6),
                      "destination": {"city": rng.choice(CITIES), "postal_code": str(rng.randint(10000, 99999))}}
            fn = function("ship_parcel", "Ship a parcel.",
                          {"parcel": prop("dict", "The parcel.", properties={
                              "tracking_code": prop("string", "Tracking code."),
                              "destination": prop("dict", "Where the parcel goes.", properties={
                                  "city": prop("string", "City."), "postal_code": prop("string", "Postal code.")},
                                  required=["city", "postal_code"])}, required=["tracking_code", "destination"])},
                          ["parcel"])
            d = parcel["destination"]
            text = f"Ship parcel {parcel['tracking_code']} to {d['city']}, postal code {d['postal_code']}."
            answer = {"ship_parcel": {"parcel": [{"tracking_code": [parcel["tracking_code"]],
                                                  "destination": [d]}]}}
            cases.append(case(level, text, [fn], answer, {"name": fn["name"], "arguments": {"parcel": parcel}}))
        elif level == "depth_3" and alt == 1:
            event = {"title": rng.choice(EVENTS), "venue": {
                "name": rng.choice(VENUES),
                "address": {"street": f"{rng.randint(1, 99)} {rng.choice(STREETS)}", "city": rng.choice(CITIES)}}}
            fn = function("create_event", "Create an event.",
                          {"event": prop("dict", "The event.", properties={
                              "title": prop("string", "Title of the event."),
                              "venue": prop("dict", "Where it takes place.", properties={
                                  "name": prop("string", "Name of the venue."),
                                  "address": prop("dict", "Address of the venue.", properties={
                                      "street": prop("string", "Street and number."), "city": prop("string", "City.")},
                                      required=["street", "city"])}, required=["name", "address"])},
                              required=["title", "venue"])}, ["event"])
            v = event["venue"]
            text = (f"Create an event called \"{event['title']}\" at {v['name']}, {v['address']['street']}, "
                    f"{v['address']['city']}.")
            answer = {"create_event": {"event": [{"title": [event["title"]], "venue": [v]}]}}
            cases.append(case(level, text, [fn], answer, {"name": fn["name"], "arguments": {"event": event}}))
        else:  # depth_3, alt 0
            account = {"account_id": code(rng, "ACC", 6), "holder": {
                "name": rng.choice(PEOPLE),
                "address": {"city": rng.choice(CITIES), "postal_code": str(rng.randint(10000, 99999))}}}
            fn = function("open_account", "Open a bank account.",
                          {"account": prop("dict", "The account.", properties={
                              "account_id": prop("string", "ID of the new account."),
                              "holder": prop("dict", "The account holder.", properties={
                                  "name": prop("string", "Full name."),
                                  "address": prop("dict", "Home address.", properties={
                                      "city": prop("string", "City."), "postal_code": prop("string", "Postal code.")},
                                      required=["city", "postal_code"])}, required=["name", "address"])},
                              required=["account_id", "holder"])}, ["account"])
            h = account["holder"]
            text = (f"Open account {account['account_id']} for {h['name']}, who lives in {h['address']['city']}, "
                    f"postal code {h['address']['postal_code']}.")
            answer = {"open_account": {"account": [{"account_id": [account["account_id"]], "holder": [h]}]}}
            cases.append(case(level, text, [fn], answer, {"name": fn["name"], "arguments": {"account": account}}))
    return cases


QUEUES = ["billing", "technical support", "sales", "returns"]
SHIPPING = ["overnight", "two-day", "standard", "economy"]
DEPARTMENTS = ["finance", "legal", "marketing", "research", "design", "security", "logistics", "training",
               "purchasing", "facilities", "customer care", "quality"]
CATEGORIES = ["garden tools", "kitchenware", "board games", "camping gear", "office chairs", "pet food", "baby toys",
              "bath towels", "phone cases", "car parts", "yoga mats", "wall clocks", "rain jackets", "desk lamps",
              "picture frames", "bicycle helmets", "coffee makers", "bed sheets", "paint brushes", "swim goggles",
              "hiking boots", "gift cards", "candles", "rugs", "headphones", "printers", "backpacks", "mirrors",
              "sunglasses", "watches"]


def enum(rng):
    cases = []
    for i, level in enumerate(levels([("4_codes", 8), ("12_codes", 9), ("30_codes", 8)])):
        alt = i % 2
        if level == "4_codes":
            meanings, prefix = (QUEUES, "Q") if alt == 0 else (SHIPPING, "S")
            codes = [f"{prefix}{k}" for k in range(1, 5)]
        elif level == "12_codes":
            meanings, prefix = (DEPARTMENTS, "D") if alt == 0 else (rng.sample(CITIES, 12), "WH")
            codes = [f"{prefix}{k:02d}" for k in range(1, 13)]
        else:
            meanings, prefix = (CATEGORIES, "C") if alt == 0 else (CITIES, "R")
            codes = [f"{prefix}{k:02d}" for k in range(1, 31)]
        shuffled = rng.sample(meanings, len(meanings))   # a new code-to-meaning map for every question
        table = ", ".join(f"{c} = {m}" for c, m in zip(codes, shuffled))
        target = rng.randrange(len(codes))
        want_code, want = codes[target], shuffled[target]
        if level == "4_codes" and alt == 0:
            item = code(rng, "TCK")
            fn = function("route_ticket", "Send a support ticket to a queue.",
                          {"ticket_id": prop("string", "ID of the ticket."),
                           "queue": prop("string", f"Queue code: {table}.", enum=codes)}, ["ticket_id", "queue"])
            text = f"Send ticket {item} to the {want} queue."
            accepted = {"ticket_id": item, "queue": want_code}
        elif level == "4_codes":
            item = code(rng, "ORD")
            fn = function("set_shipping_service", "Choose the shipping service for an order.",
                          {"order_id": prop("string", "ID of the order."),
                           "service": prop("string", f"Service code: {table}.", enum=codes)}, ["order_id", "service"])
            text = f"Ship order {item} with {want} shipping."
            accepted = {"order_id": item, "service": want_code}
        elif level == "12_codes" and alt == 0:
            person = rng.choice(PEOPLE)
            fn = function("assign_department", "Assign an employee to a department.",
                          {"employee_name": prop("string", "Full name of the employee."),
                           "department": prop("string", f"Department code: {table}.", enum=codes)},
                          ["employee_name", "department"])
            text = f"Assign {person} to the {want} department."
            accepted = {"employee_name": person, "department": want_code}
        elif level == "12_codes":
            item = code(rng, "PAL")
            fn = function("move_pallet", "Move a pallet to a warehouse.",
                          {"pallet_id": prop("string", "ID of the pallet."),
                           "warehouse": prop("string", f"Warehouse code: {table}.", enum=codes)},
                          ["pallet_id", "warehouse"])
            text = f"Move pallet {item} to the {want} warehouse."
            accepted = {"pallet_id": item, "warehouse": want_code}
        elif alt == 0:
            item = code(rng, "PRD")
            fn = function("list_product", "List a product in a category of the online shop.",
                          {"product_id": prop("string", "ID of the product."),
                           "category": prop("string", f"Category code: {table}.", enum=codes)},
                          ["product_id", "category"])
            text = f"List product {item} in the {want} category."
            accepted = {"product_id": item, "category": want_code}
        else:
            item = code(rng, "STR")
            fn = function("set_store_region", "Set the sales region of a store.",
                          {"store_id": prop("string", "ID of the store."),
                           "region": prop("string", f"Region code: {table}.", enum=codes)}, ["store_id", "region"])
            text = f"Put store {item} in the {want} region."
            accepted = {"store_id": item, "region": want_code}
        cases.append(simple(level, text, fn, accepted))
    return cases


def number(rng):
    cases = []
    for i, level in enumerate(levels([("integer", 8), ("float", 8), ("integer_and_float", 9)])):
        alt = i % 4
        if level == "integer" and alt in (0, 1):
            sku = code(rng, "SKU", 3)
            qty = rng.randint(1000, 99999) if alt == 0 else rng.randint(13, 99)
            shown = f"{qty:,}" if alt == 0 else words(qty)
            fn = function("set_stock_level", "Set the number of units in stock.",
                          {"sku": prop("string", "Product code."), "quantity": prop("integer", "Units in stock.")},
                          ["sku", "quantity"])
            cases.append(simple(level, f"Set the stock level of {sku} to {shown} units.", fn,
                                {"sku": sku, "quantity": qty}))
        elif level == "integer":
            member, change = code(rng, "MBR"), -rng.randint(5, 500) if alt == 2 else rng.randint(5, 500)
            fn = function("adjust_points", "Change a member's loyalty points by a whole number.",
                          {"member_id": prop("string", "ID of the member."),
                           "change": prop("integer", "Points to add; negative to remove.")}, ["member_id", "change"])
            cases.append(simple(level, f"Change the points of member {member} by {change}.", fn,
                                {"member_id": member, "change": change}))
        elif level == "float" and alt in (0, 1):
            zone = rng.choice(["A", "B", "C", "D", "E"])
            temp = rng.choice([-1, 1]) * rng.randint(50, 2600) / 100
            if temp == int(temp):
                temp += 0.25
            fn = function("set_thermostat", "Set the target temperature of a zone.",
                          {"zone": prop("string", "Zone letter."),
                           "temperature_c": prop("float", "Target temperature in degrees Celsius.")},
                          ["zone", "temperature_c"])
            cases.append(simple(level, f"Set the thermostat in zone {zone} to {temp} degrees Celsius.", fn,
                                {"zone": zone, "temperature_c": temp}))
        elif level == "float":
            sku = code(rng, "SKU", 3)
            price = rng.randint(1000, 250000) / 100 if alt == 2 else rng.randint(1, 99) / 1000
            if price == int(price):
                price += 0.5
            shown = f"{price:,.2f}" if alt == 2 else f"{price}"
            fn = function("set_price", "Set the price of a product in US dollars.",
                          {"sku": prop("string", "Product code."), "price": prop("float", "Price in US dollars.")},
                          ["sku", "price"])
            cases.append(simple(level, f"Change the price of {sku} to {shown} dollars.", fn,
                                {"sku": sku, "price": price}))
        elif alt in (0, 2):
            sku, qty = code(rng, "SKU", 3), rng.randint(2, 40)
            unit = rng.randint(50, 9999) / 100
            if unit == int(unit):
                unit += 0.25
            fn = function("add_line_item", "Add a line to an invoice.",
                          {"sku": prop("string", "Product code."), "quantity": prop("integer", "Number of units."),
                           "unit_price": prop("float", "Price of one unit.")}, ["sku", "quantity", "unit_price"])
            cases.append(simple(level, f"Add {qty} units of {sku} to the invoice at {unit:.2f} each.", fn,
                                {"sku": sku, "quantity": qty, "unit_price": unit}))
        else:
            km, minutes = rng.randint(30, 425) / 10, rng.randint(15, 240)
            if km == int(km):
                km += 0.5
            fn = function("log_run", "Log a run.",
                          {"distance_km": prop("float", "Distance in kilometers."),
                           "duration_minutes": prop("integer", "Duration in whole minutes.")},
                          ["distance_km", "duration_minutes"])
            cases.append(simple(level, f"Log a run of {km} km that took {minutes} minutes.", fn,
                                {"distance_km": km, "duration_minutes": minutes}))
    return cases


# (name, type, default, description, values to draw from, how the question says it)
SEARCH_OPTIONAL = [
    ("max_results", "integer", 10, "Maximum number of results.", [5, 20, 25, 50], "showing at most {} results"),
    ("min_rating", "integer", 0, "Lowest star rating to include.", [3, 4], "rated at least {} stars"),
    ("in_stock_only", "boolean", False, "Only items in stock.", [True], "only items in stock"),
    ("brand", "string", "any", "Brand name.", ["Acme", "Norda", "Vexo", "Lumen"], "from the brand {}"),
    ("max_price", "integer", 1000, "Highest price in dollars.", [50, 120, 300], "under {} dollars"),
    ("sort_by", "string", "relevance", "Sort order.", ["price", "rating", "newest"], "sorted by {}"),
    ("page", "integer", 1, "Page of results.", [2, 3, 4], "showing page {} of the results"),
    ("include_ads", "boolean", True, "Whether to include sponsored items.", [False], "without sponsored items"),
]
MEETING_OPTIONAL = [
    ("duration_minutes", "integer", 30, "Length in minutes.", [45, 60, 90], "for {} minutes"),
    ("location", "string", "online", "Room or place.", ["Room 4B", "Room 12", "Lab 2"], "in {}"),
    ("reminder_minutes", "integer", 10, "Reminder before the start, in minutes.", [5, 15, 30],
     "with a reminder {} minutes before"),
    ("recurring", "boolean", False, "Whether it repeats every week.", [True], "repeating every week"),
    ("agenda", "string", "TBD", "Agenda.", ["Budget", "Hiring plan", "Roadmap"], 'with the agenda "{}"'),
    ("max_attendees", "integer", 10, "Largest number of attendees.", [4, 6, 20], "for at most {} people"),
    ("send_invites", "boolean", True, "Whether to email invitations.", [False], "without sending invitations"),
    ("timezone", "string", "UTC", "Time zone.", ["Asia/Dhaka", "Europe/Paris", "America/Lima"], "in the {} time zone"),
]


def optional(rng):
    cases = []
    for i, level in enumerate(levels([("2_optional", 8), ("4_optional", 9), ("8_optional", 8)])):
        k = int(level[0])
        mentioned_n = i % 3
        if i % 2 == 0:
            name, desc, pool = "search_products", "Search the shop's products.", SEARCH_OPTIONAL
            query = rng.choice(["wireless mouse", "running shoes", "desk lamp", "water bottle", "travel mug"])
            required = {"query": (prop("string", "What to search for."), query)}
            opening = f'Search the shop for "{query}"'
        else:
            name, desc, pool = "schedule_meeting", "Schedule a meeting.", MEETING_OPTIONAL
            title, day = rng.choice(SUBJECTS), date(rng)
            required = {"title": (prop("string", "Title of the meeting."), title),
                        "date": (prop("string", "Date, YYYY-MM-DD."), day)}
            opening = f'Schedule a meeting called "{title}" on {day}'
        chosen = rng.sample(pool, k)
        mentioned = chosen[:mentioned_n]
        properties = {p: spec for p, (spec, _) in required.items()}
        accepted, answer = {p: v for p, (_, v) in required.items()}, {p: [v] for p, (_, v) in required.items()}
        phrases = []
        for pname, ptype, default, pdesc, values, phrase in chosen:
            properties[pname] = prop(ptype, f"{pdesc} Default: {json.dumps(default)}.", default=default)
            if (pname, ptype, default, pdesc, values, phrase) in mentioned:
                value = rng.choice(values)
                answer[pname] = [value]
                accepted[pname] = value
                phrases.append(phrase.format(value))
            else:
                answer[pname] = ["", default]
        fn = function(name, desc, properties, list(required))
        text = opening + (", " + " and ".join(phrases) if phrases else "") + "."
        cases.append(case(level, text, [fn], {name: answer}, {"name": name, "arguments": accepted}))
    return cases


LATIN = {"names": ["José Núñez", "Zoë Lefèvre", "Søren Ødegård", "Łukasz Wójcik", "Ana Conceição", "Jürgen Groß",
                   "Çağrı Yılmaz", "Đorđe Petrović", "Agnès Dupré", "Hélène Brücke"],
         "cities": ["Zürich", "Kraków", "Málaga", "Besançon", "Tromsø", "Reykjavík", "Düsseldorf", "São Paulo",
                    "Göteborg", "Łódź"],
         "titles": ["Fête de la Musique", "Café Crème Night", "Noël Market", "Ålesund Jazz Week", "Señorita Fiesta",
                    "Über Tech Day"]}
NON_LATIN = [("Иван Петров", "Москва", "Русский вечер"), ("Αλέξης Παππάς", "Αθήνα", "Θερινό σινεμά"),
             ("王伟", "北京", "春节晚会"), ("佐藤花子", "東京", "夏祭り"), ("김민준", "서울", "가을 음악회"),
             ("محمد علي", "القاهرة", "ليلة الشعر"), ("דוד כהן", "תל אביב", "ערב ג'אז"),
             ("राहुल शर्मा", "मुंबई", "दिवाली मेला"), ("সাকিব হাসান", "ঢাকা", "বৈশাখী মেলা")]
EMOJI_TEXTS = ["Happy birthday 🎉🎂", "See you at 8 ☕", "Great job 👍 thanks!", "Meeting moved ➡️ Friday",
               "Ça va? 😊", "Привет 👋", "生日快乐 🎈", "All done ✅", "Good luck 🍀 today", "Lunch at 1 🍕?"]


def unicode(rng):
    cases = []
    people, texts, non_latin = Deck(rng, PEOPLE), Deck(rng, EMOJI_TEXTS), Deck(rng, NON_LATIN)
    names, cities, titles = (Deck(rng, LATIN[k]) for k in ("names", "cities", "titles"))
    for i, level in enumerate(levels([("latin_accents", 8), ("non_latin", 9), ("emoji", 8)])):
        alt = i % 2
        if level == "emoji":
            name, text_value = people.draw(), texts.draw()
            fn = function("send_message", "Send a chat message.",
                          {"recipient": prop("string", "Name of the recipient."),
                           "text": prop("string", "The message, exactly as given.")}, ["recipient", "text"])
            cases.append(simple(level, f'Send the message "{text_value}" to {name}.', fn,
                                {"recipient": name, "text": text_value}))
            continue
        if level == "latin_accents":
            name, city, title = names.draw(), cities.draw(), titles.draw()
        else:
            name, city, title = non_latin.draw()
        if alt == 0:
            fn = function("add_contact", "Add a person to the address book.",
                          {"name": prop("string", "Full name, exactly as given."),
                           "city": prop("string", "City, exactly as given.")}, ["name", "city"])
            cases.append(simple(level, f"Add {name} from {city} to my contacts.", fn, {"name": name, "city": city}))
        else:
            fn = function("create_event", "Create a calendar event.",
                          {"title": prop("string", "Title, exactly as given."),
                           "location": prop("string", "Place, exactly as given.")}, ["title", "location"])
            cases.append(simple(level, f'Create an event called "{title}" in {city}.', fn,
                                {"title": title, "location": city}))
    return cases


OBJECTS = {"invoice": "INV", "order": "ORD", "ticket": "TCK", "customer": "CUS", "shipment": "SHP", "refund": "REF",
           "coupon": "CPN", "employee": "EMP", "meeting": "MTG", "project": "PRJ", "contract": "CTR", "booking": "BKG",
           "payment": "PAY", "report": "RPT", "subscription": "SUB", "account": "ACC"}
ACTIONS = ["get", "delete", "archive", "restore", "cancel", "approve", "duplicate", "lock"]
MENU_SIZE = (10, 5)   # objects x actions = 50 functions


def article(word):
    return "an" if word[0] in "aeiou" else "a"


def menu(rng):
    cases = []
    ranges = {"position_1_10": (0, 9), "position_21_30": (20, 29), "position_41_50": (40, 49)}
    for i, level in enumerate(levels([("position_1_10", 8), ("position_21_30", 9), ("position_41_50", 8)])):
        objects = rng.sample(sorted(OBJECTS), MENU_SIZE[0])
        actions = rng.sample(ACTIONS, MENU_SIZE[1])
        target_object, target_action = rng.choice(objects), rng.choice(actions)
        functions = [function(f"{a}_{o}", f"{a.capitalize()} {article(o)} {o}.",
                              {f"{o}_id": {"type": "string"}}, [f"{o}_id"]) for o in objects for a in actions]
        rng.shuffle(functions)
        target = next(f for f in functions if f["name"] == f"{target_action}_{target_object}")
        functions.remove(target)
        functions.insert(rng.randint(*ranges[level]), target)
        item = code(rng, OBJECTS[target_object])
        text = rng.choice(["{A} the {o} {id}.", "Please {a} {o} {id}.", "Can you {a} the {o} with ID {id}?"]).format(
            A=target_action.capitalize(), a=target_action, o=target_object, id=item)
        answer = {target["name"]: {f"{target_object}_id": [item]}}
        cases.append(case(level, text, functions, answer,
                          {"name": target["name"], "arguments": {f"{target_object}_id": item}}))
    return cases


MAKERS = {"stress_plain": plain, "stress_nested": nested, "stress_enum": enum, "stress_number": number,
          "stress_optional": optional, "stress_unicode": unicode, "stress_menu": menu}


def build():
    """Returns {group: [(question entry, answer entry, reference call), ...]}."""
    suite = {}
    for group in GROUPS:
        rng = random.Random(f"{group}-{SEED}")   # one generator per group, so groups do not affect each other
        rows = []
        for n, c in enumerate(MAKERS[group](rng)):
            qid = f"{group}_{n}"
            rows.append(({"id": qid, "level": c["level"], "question": [[{"role": "user", "content": c["text"]}]],
                          "function": c["functions"]},
                         {"id": qid, "ground_truth": [c["answer"]]}, c["reference"]))
        suite[group] = rows
    return suite


def lines(entries):
    return "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)


def write(suite):
    (OUT / "possible_answer").mkdir(parents=True, exist_ok=True)
    for group, rows in suite.items():
        (OUT / f"{group}.json").write_text(lines(r[0] for r in rows), encoding="utf-8", newline="\n")
        (OUT / "possible_answer" / f"{group}.json").write_text(lines(r[1] for r in rows), encoding="utf-8",
                                                                newline="\n")
    print(f"Wrote {sum(len(r) for r in suite.values())} questions in {len(suite)} groups to {OUT}")


# ---------------------------------------------------------------- checks
def leaves(value):
    """All scalar values inside a (nested) argument value."""
    if isinstance(value, dict):
        return [x for v in value.values() for x in leaves(v)]
    if isinstance(value, list):
        return [x for v in value for x in leaves(v)]
    return [value]


def mutate(value, last=False):
    """A wrong version of an argument value: its first (or last, i.e. deepest here) scalar is changed."""
    if isinstance(value, dict):
        key = list(value)[-1 if last else 0]
        return {**value, key: mutate(value[key], last)}
    if isinstance(value, list):
        return value[:-1] + [mutate(value[-1], last)] if last else [mutate(value[0], last)] + value[1:]
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.5
    return value + "x"


def check():
    """Checks that the files on disk are exactly what build() makes, and that every answer key is sound:

    1. 25 questions per group, unique ids.
    2. The reference call is accepted by BFCL's checker (scored the same way as src/score.py) and matches the
       JSON schema that constrained decoding uses (src/schema.py).
    3. A wrong call is rejected: one value changed (enum: another code from the list; menu: another action on the
       same object). For the optional group, a call that also sets an argument the question did not mention is
       rejected too.
    4. Every string in the reference call is written in the question, except enum codes.
    """
    from jsonschema import Draft202012Validator

    from data import load_answer_key, load_questions
    from schema import calls_schema
    from score import score_reply

    suite = build()
    for group, rows in suite.items():
        for path, entries in ((OUT / f"{group}.json", [r[0] for r in rows]),
                              (OUT / "possible_answer" / f"{group}.json", [r[1] for r in rows])):
            on_disk = path.read_bytes().decode("utf-8").replace("\r\n", "\n")   # Git on Windows may add \r
            assert on_disk == lines(entries), f"{path} differs from what the generator makes"
    print("Files on disk are exactly what the generator makes.")

    counts, rejected = Counter(), Counter()
    ids = [r[0]["id"] for rows in suite.values() for r in rows]
    assert len(ids) == len(set(ids)) == PER_GROUP * len(GROUPS)
    for group, rows in suite.items():
        questions = {q["id"]: q for q in load_questions(group)}
        keys = load_answer_key(group)
        assert len(questions) == len(keys) == PER_GROUP, group
        for entry, _, reference in rows:
            qid, q, key = entry["id"], questions[entry["id"]], keys[entry["id"]]
            reply = json.dumps([reference], ensure_ascii=False)
            result = score_reply(group, reply, key)
            assert result["valid"], (qid, result["error"])
            assert Draft202012Validator(calls_schema(q["functions"])).is_valid([reference]), qid

            fn = next(f for f in q["functions"] if f["name"] == reference["name"])
            wrong = []
            if group == "stress_menu":
                obj = reference["name"].split("_", 1)[1]
                other = next(f["name"] for f in q["functions"] if f["name"].endswith("_" + obj)
                             and f["name"] != reference["name"])
                wrong.append({"name": other, "arguments": reference["arguments"]})
            else:
                first = fn["parameters"]["required"][0] if group != "stress_enum" else fn["parameters"]["required"][1]
                spec = fn["parameters"]["properties"][first]
                value = reference["arguments"][first]
                bads = [next(c for c in spec["enum"] if c != value)] if "enum" in spec else [mutate(value)]
                if isinstance(value, (dict, list)):
                    bads.append(mutate(value, last=True))   # the deepest value too
                wrong += [{"name": reference["name"], "arguments": {**reference["arguments"], first: b}} for b in bads]
            if group == "stress_optional":
                unset = [p for p, s in fn["parameters"]["properties"].items() if p not in reference["arguments"]]
                if unset:
                    p = unset[0]
                    wrong.append({"name": reference["name"],
                                  "arguments": {**reference["arguments"], p: mutate(fn["parameters"]["properties"][p]["default"])}})
            for w in wrong:
                assert not score_reply(group, json.dumps([w], ensure_ascii=False), key)["valid"], (qid, w)
                rejected[group] += 1

            text = q["messages"][0]["content"]
            enum_params = {p for p, s in fn["parameters"]["properties"].items() if "enum" in s}
            for p, v in reference["arguments"].items():
                if p in enum_params:
                    continue
                for leaf in leaves(v):
                    if isinstance(leaf, str):
                        assert leaf in text, (qid, leaf)
            counts[group, entry["level"]] += 1

    for group in GROUPS:
        levels_ = ", ".join(f"{level} {n}" for (g, level), n in counts.items() if g == group)
        print(f"{group:16s} {sum(n for (g, _), n in counts.items() if g == group):3d} questions ({levels_}); "
              f"{rejected[group]} wrong calls rejected")
    print("All checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="check the files on disk instead of writing them")
    args = parser.parse_args()
    if args.check:
        sys.path.insert(0, str(REPO / "src"))
        check()
    else:
        write(build())


if __name__ == "__main__":
    main()
