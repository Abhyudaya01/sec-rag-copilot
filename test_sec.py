from sec_edgar_api import EdgarClient

# Initialize SEC client (replace with your info)
edgar = EdgarClient(
    user_agent="YourName yourname@email.com"  # Change this to your actual info
)

print("Testing SEC EDGAR API...")

try:
    # Get Apple's company info
    data = edgar.get_submissions(cik="320193")
    
    print(f"\n✅ Success! Found company: {data['name']}")
    print(f"Ticker: {data['tickers'][0]}")
    print(f"Total filings available: {len(data['filings']['recent']['form'])}")
    
    # Show recent 10-K filings
    recent_10k = []
    for i, form in enumerate(data['filings']['recent']['form']):
        if form == '10-K' and len(recent_10k) < 3:
            recent_10k.append({
                'date': data['filings']['recent']['filingDate'][i],
                'form': form
            })
    
    print("\nRecent 10-K filings:")
    for filing in recent_10k:
        print(f"  - {filing['form']} filed on {filing['date']}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
