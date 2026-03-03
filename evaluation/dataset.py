# 25 hand-crafted Q&A pairs for RAGAS evaluation.
# Split across all three CRAG paths — 8 CORRECT, 9 AMBIGUOUS, 8 INCORRECT.

EVAL_DATASET = [

    # CORRECT PATH — 8 questions
    # Answers are directly and fully in uae_data.txt
    # Expected: at least one chunk scores > 0.7 (UPPER_TH)
    
    {
        "question":  "What is the corporate tax rate for tech startups in DIFC with revenue under 3 million AED?",
        "reference": "The corporate tax rate for tech startups in DIFC with revenue under 3 million AED is 0%.",
        "path":      "correct"
    },
    {
        "question":  "When must DIFC AI startups appoint a Data Ethics Officer?",
        "reference": "All AI startups in DIFC must appoint a Data Ethics Officer by Q3 2026.",
        "path":      "correct"
    },
    {
        "question":  "What are the working hours at the end of the week for DIFC companies?",
        "reference": "The standard work week for DIFC companies is Monday to Friday, 4.5 days, ending at 12:00 PM on Friday.",
        "path":      "correct"
    },
    {
        "question":  "What Golden Visa is available for AI engineers in the UAE?",
        "reference": "AI Engineers with 2 or more years of professional experience are eligible for the 10-year Golden Visa immediately through the DIFC pathway.",
        "path":      "correct"
    },
    {
        "question":  "What is the DEWS scheme minimum contribution for employees with less than five years of service?",
        "reference": "The minimum DEWS contribution is 5.83% of basic salary for employees with less than five years of service.",
        "path":      "correct"
    },
    {
        "question":  "What percentage of foreign ownership does DIFC allow for companies?",
        "reference": "DIFC offers 100% foreign ownership without the need for a local UAE partner.",
        "path":      "correct"
    },
    {
        "question":  "What is the maximum probation period allowed under DIFC Employment Law?",
        "reference": "The probation period under DIFC Employment Law is capped at six months.",
        "path":      "correct"
    },
    {
        "question":  "What was the new company registration growth rate in DIFC in the first half of 2025?",
        "reference": "DIFC registered 1,081 new active companies between January and June 2025, a 32% increase compared to the same period in 2024.",
        "path":      "correct"
    },

    
    # AMBIGUOUS PATH — 9 questions
    # One part of the question is in uae_data.txt, the other is not
    # Expected: no chunk > 0.7, but some > 0.3 (AMBIGUOUS)
    
    {
        "question":  "What is the DIFC corporate tax rate for tech startups and what is the current UAE dirham to US dollar exchange rate?",
        "reference": "The corporate tax rate for DIFC tech startups with revenue under 3M AED is 0%. The UAE dirham is pegged to the US dollar at approximately 3.67 AED per USD.",
        "path":      "ambiguous"
    },
    {
        "question":  "What visa options exist for AI engineers in the UAE and how does the UAE Golden Visa compare to the Singapore Employment Pass?",
        "reference": "UAE offers a 10-year Golden Visa for AI engineers with 2+ years experience through DIFC. Singapore's Employment Pass is an employer-tied work visa with different eligibility and duration requirements.",
        "path":      "ambiguous"
    },
    {
        "question":  "What are the working hours rules in DIFC and what are typical working hours at tech companies in San Francisco?",
        "reference": "DIFC mandates a maximum of 48 hours per week with the work week ending at noon Friday. Tech companies in San Francisco typically follow a 40-hour work week with flexible arrangements common in the tech industry.",
        "path":      "ambiguous"
    },
    {
        "question":  "What crypto regulations apply in DIFC and what is the current price of Bitcoin?",
        "reference": "DIFC updated its crypto token regulations effective January 2026, requiring firms to assess tokens themselves. Bitcoin price fluctuates and must be checked on a live financial source.",
        "path":      "ambiguous"
    },
    {
        "question":  "What is the DIFC standard corporate tax rate and how does it compare to the corporate tax rate in Singapore?",
        "reference": "The standard UAE corporate tax rate is 9% on income above AED 375,000, with 0% for qualifying free zone persons. Singapore's corporate tax rate is 17% with various incentive schemes.",
        "path":      "ambiguous"
    },
    {
        "question":  "What employee savings scheme exists in DIFC and what are the latest global trends in employee retirement benefits?",
        "reference": "DIFC operates the DEWS scheme requiring employer contributions of 5.83% for first five years rising to 8.33% thereafter. Globally, defined contribution plans are increasingly replacing defined benefit pensions.",
        "path":      "ambiguous"
    },
    {
        "question":  "What AI regulations are in place in DIFC and what are the latest developments in AI regulation in the European Union?",
        "reference": "DIFC requires AI companies to appoint a Data Ethics Officer by Q3 2026 and comply with DFSA data protection rules. The EU AI Act is the primary AI regulatory framework in Europe.",
        "path":      "ambiguous"
    },
    {
        "question":  "What are the new VCC regulations in DIFC and what is a variable capital company used for in other jurisdictions like Singapore?",
        "reference": "DIFC enacted VCC Regulations on 9 February 2026, allowing flexible corporate structures without qualifying purpose restrictions. Singapore introduced its VCC framework in 2020 as a vehicle for investment funds.",
        "path":      "ambiguous"
    },
    {
        "question":  "What annual leave entitlement do DIFC employees receive and how does this compare to annual leave entitlements in the United Kingdom?",
        "reference": "DIFC employees are entitled to 30 calendar days of annual leave after one year of service. In the UK, the statutory minimum is 28 days including public holidays.",
        "path":      "ambiguous"
    },

    
    # INCORRECT PATH — 8 questions
    # Completely outside uae_data.txt — full Tavily fallback
    # Expected: all chunks score < 0.3 (INCORRECT)
    
    {
        "question":  "Who won the 2024 United States Presidential election?",
        "reference": "Donald Trump won the 2024 US Presidential election, defeating Kamala Harris with 312 electoral college votes.",
        "path":      "incorrect"
    },
    {
        "question":  "What is the current price of gold per ounce in US dollars?",
        "reference": "Gold price fluctuates daily. As of early 2026, gold has been trading above $2,600 per troy ounce. Check a live financial source for the current price.",
        "path":      "incorrect"
    },
    {
        "question":  "What is the latest version of the iPhone released by Apple?",
        "reference": "Apple released the iPhone 16 series in September 2024. Check Apple's website for the most current product lineup.",
        "path":      "incorrect"
    },
    {
        "question":  "What is the current stock price of NVIDIA?",
        "reference": "NVIDIA stock price changes continuously during trading hours. Check a financial source like Yahoo Finance for the current price.",
        "path":      "incorrect"
    },
    {
        "question":  "What is the population of Dubai in 2025?",
        "reference": "Dubai's population in 2025 is approximately 3.6 million people, making it the most populous city in the UAE.",
        "path":      "incorrect"
    },
    {
        "question":  "Which team won the FIFA World Cup in 2022?",
        "reference": "Argentina won the 2022 FIFA World Cup, defeating France on penalties in the final held in Qatar.",
        "path":      "incorrect"
    },
    {
        "question":  "What is the current interest rate set by the US Federal Reserve?",
        "reference": "The US Federal Reserve sets interest rates at FOMC meetings throughout the year. Check the Federal Reserve website or financial news for the current federal funds rate.",
        "path":      "incorrect"
    },
    {
        "question":  "What is the latest Claude model released by Anthropic?",
        "reference": "Anthropic releases Claude models periodically. As of early 2026, the Claude 3.5 and Claude 4 series models are available. Check Anthropic's website for the most current model lineup.",
        "path":      "incorrect"
    }]