"""
SEC filing downloader and parser — Full Phase 2 code
"""

from sec_edgar_api import EdgarClient
import json
import requests
from bs4 import BeautifulSoup
import os

class SECDownloader:
    def __init__(self, user_agent):
        """
        Initialize SEC downloader with a user agent string.
        Args:
            user_agent (str): Your name and email, e.g. "Your Name your.email@example.com"
        """
        self.edgar = EdgarClient(user_agent=user_agent)

    def get_company_filings(self, cik, form_types=['10-K', '10-Q']):
        """
        Retrieve recent filings metadata for a company by CIK.
        Returns: list of dicts containing filings metadata.
        """
        data = self.edgar.get_submissions(cik=cik)
        filings = []
        for i, form in enumerate(data['filings']['recent']['form']):
            if form in form_types:
                filings.append({
                    'company': data['name'],
                    'ticker': data['tickers'][0] if data['tickers'] else 'N/A',
                    'cik': cik,
                    'form': form,
                    'filing_date': data['filings']['recent']['filingDate'][i],
                    'accession_number': data['filings']['recent']['accessionNumber'][i],
                    'primary_document': data['filings']['recent']['primaryDocument'][i]
                })
        return filings

    def save_filings_list(self, cik, output_file='data/filings_list.json'):
        """
        Save list of filings metadata as JSON file.
        """
        filings = self.get_company_filings(cik)
        with open(output_file, 'w') as f:
            json.dump(filings, f, indent=2)
        print(f"Saved {len(filings)} filings into {output_file}")
        return filings


def fetch_filing_document(cik, accession_number, primary_document, user_agent):
    """
    Download the filing HTML document content from SEC EDGAR Archives.
    Returns: Raw HTML content of filing.
    """
    base_url = "https://www.sec.gov/Archives/edgar/data"
    cik = cik.lstrip('0')  # Remove leading zeros if present
    url = f"{base_url}/{cik}/{accession_number.replace('-', '')}/{primary_document}"
    response = requests.get(url, headers={"User-Agent": user_agent})
    if response.status_code != 200:
        raise Exception(f"Failed to fetch filing document: {url}, status code: {response.status_code}")
    return response.text


def extract_text_from_html(html):
    """
    Extract clean text from filing HTML content.
    Returns: Cleaned plain text extracted from HTML.
    """
    soup = BeautifulSoup(html, 'html.parser')
    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()
    text = ' '.join(soup.get_text(separator=' ').split())
    return text


if __name__ == "__main__":
    # ----- Step 0: User settings -----
    USER_AGENT = "Abhyuday Lohani abhyudaylohani@gmail.com"
    CIK = "0000320193"  # Apple Inc CIK (example)

    downloader = SECDownloader(user_agent=USER_AGENT)

    # Step 1: Get latest filings metadata
    filings = downloader.get_company_filings(CIK)
    print(f"Found {len(filings)} filings for company {filings[0]['company'] if filings else 'N/A'}:")
    for filing in filings[:5]:
        print(f"  - {filing['form']} filed on {filing['filing_date']}")

    # Step 2: Fetch and extract the first filing's content
    if filings:
        filing = filings[0]
        print(f"\nFetching filing document for {filing['form']} dated {filing['filing_date']} ...")
        html_content = fetch_filing_document(
            filing['cik'], filing['accession_number'], filing['primary_document'], USER_AGENT
        )
        print("Extracting text from HTML...")
        text = extract_text_from_html(html_content)
        print(f"Extracted text length: {len(text)} characters")
        print(f"Sample text preview:\n{text[:1000]}")

        # Step 3: Save cleaned text for Phase 3 chunking
        os.makedirs("data/processed", exist_ok=True)
        with open("data/processed/extracted_text.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("Extracted filing text saved to data/processed/extracted_text.txt")
    else:
        print("No filings found.")
