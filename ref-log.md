Implementing the multi-agent travel planner taught me a lot in a hands-on way.

Main takeaways:
- Split roles helped: the Planner writes the day-by-day plan (activities, times, places, rough costs, logistics). The Reviewer checks facts and finds problems. This division made each agent focused.
- Prompts matter: telling the Planner to use a fixed format (Trip Overview, Day 1..., Budget, Logistics) made it easy for the Reviewer to check specific parts. Asking the Reviewer to give a Validation Summary, Delta List, and Final Recommendation made its output useful and clear.

Problems I hit and how I handled them:
- Planner has no internet: it can only guess opening times and prices. I made the Planner conservative — it gives ranges and notes assumptions so the Reviewer can check them.
- Tool use balance: the Reviewer needs to fact-check but should not spam the web. I told it to use short, targeted searches (like "X opening hours city" or "train time A to B") and to cite one short line. That kept searches useful and not excessive.

Design choices and extra ideas:
- Personas: Planner = creative, budget-aware travel advisor. Reviewer = cautious operations checker. They complement each other: Planner is bold, Reviewer keeps it realistic.
- Delta List is key: concrete numbered fixes (with reasons and citations) make it easy to apply changes or show them to users.

External help used:
- GPT-5 helped me write and refine prompts, debug issues, and improve code readability.
- The provided `internet_search` (Tavily) is used by the Reviewer for live checks.