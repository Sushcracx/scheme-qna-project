# -*- coding: utf-8 -*-
"""scraper.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uKVYBXYD86JKmM8YdlJHUj4F7d89J0_z
"""

import logging
import time
import pandas
from selenium.webdriver import Chrome # Import Chrome directly
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait # Explicit import
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
# Exception imports
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException
from selenium.webdriver.common.action_chains import ActionChains # Keep for potential click issues

# --- Global Settings ---
SITE_TARGET_URL = "https://www.myscheme.gov.in/search"
SCHEMES_TO_COLLECT = 100
MAX_PAGINATION_DEPTH = 15 # Limit pages to prevent runaway script
RESULT_FILENAME_CSV = "myscheme_results_v2.csv"
RESULT_FILENAME_JSON = "myscheme_results_v2.json"
REQUEST_DELAY_SECONDS = 3 # Be polite to the server
WEBDRIVER_TIMEOUT = 30 # Max wait time for elements/pages
CONSOLE_LOG_LEVEL = logging.INFO # Controls console output verbosity

# --- Element Locators (XPath and CSS) ---
# Using a dictionary for easy reference
page_locators = {
    'card': "div.mx-auto.rounded-xl.shadow-md",
    'link': "h2 > a",
    'dept': "h2.mt-3.font-normal",
    'summary': "span.line-clamp-2 > span",
    'category_tag': "div[title] > span",
    'page_num_xpath': "//ul[contains(@class, 'list-none')]//li[normalize-space(text())='{}']",
    'current_page_xpath': "//ul[contains(@class, 'list-none')]//li[contains(@class, 'bg-green-700')]",
    'detail_title_xpath': "//h1[contains(@class, 'font-bold text-xl sm:text-2xl')]",
    'benefits_content_xpath': "//div[@id='benefits']//div[contains(@class, 'markdown-options')]",
    'eligibility_content_xpath': "//div[@id='eligibility']//div[contains(@class, 'markdown-options')]",
    'process_content_xpath': "//div[@id='application-process']//div[contains(@class, 'markdown-options')]",
    'docs_content_xpath': "//div[@id='documents-required']//div[contains(@class, 'markdown-options')]",
}

# Setup application logging
logging.basicConfig(level=CONSOLE_LOG_LEVEL,
                    format='%(asctime)s | %(levelname)-8s | %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__) # Use a named logger

# --- Detail Retrieval Function ---
def retrieve_detail_info(web_driver_instance, locator_xpath):
    """Attempts to retrieve plain text content for a given detail section."""
    default_value = "Not Found"
    try:
        # Wait for the element to exist first
        wait = WebDriverWait(web_driver_instance, 10)
        content_element = wait.until(EC.presence_of_element_located((By.XPATH, locator_xpath)))
        # Extract text
        plain_text = content_element.text.strip()
        # Return text if found, otherwise indicate emptiness
        if plain_text:
            return ' '.join(plain_text.split()) # Normalize whitespace
        else:
            log.debug("Section element present but visually empty: {}".format(locator_xpath))
            return "Section Empty"
    except TimeoutException:
        # Element wasn't found within the wait time
        log.warning("Could not find detail element: {}".format(locator_xpath))
        return default_value
    except Exception as ex:
        # Catch any other exceptions during extraction
        log.error("Failed to get detail from {}: {}".format(locator_xpath, ex))
        return "Extraction Error"

# --- Main Script Body ---
def run_scraper():
    """Main function to orchestrate the scraping process."""
    scraped_data_storage = [] # List to hold all collected dictionaries
    web_driver = None # Ensure variable exists in outer scope
    start_ts = time.time()
    log.info("MyScheme Scraper (v2) Started.")

    try:
        # 1. Initialize Chrome Driver
        log.info("Initializing Chrome WebDriver via WebDriverManager...")
        options = Options()
        # options.add_argument("--headless=new") # Enable for headless execution
        options.add_argument("--start-maximized")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-extensions")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_experimental_option('excludeSwitches', ['enable-logging'])

        svc = Service(ChromeDriverManager().install())
        web_driver = Chrome(service=svc, options=options)
        log.info("WebDriver successfully initialized.")
        web_driver_wait = WebDriverWait(web_driver, WEBDRIVER_TIMEOUT)

        # 2. Load Search Page
        log.info("Loading MyScheme search page: {}".format(SITE_TARGET_URL))
        web_driver.get(SITE_TARGET_URL)
        time.sleep(4) # Static wait for initial render
        log.info("Search page loaded. Title: '{}'".format(web_driver.title))

        # 3. Get Initial Schemes (Page 1)
        log.info("Fetching schemes from page 1...")
        try:
            initial_cards = web_driver_wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, page_locators['card']))
            )
            log.info("Found {} potential schemes on first page.".format(len(initial_cards)))
            # Extract summary details (similar logic to v1)
            for i, card in enumerate(initial_cards):
                 scheme_dict = {}
                 try:
                     link_el = card.find_element(By.CSS_SELECTOR, page_locators['link'])
                     url_val = link_el.get_attribute('href')
                     scheme_dict["url"] = "{}/{}".format(BASE_URL.rstrip('/'), url_val.lstrip('/')) if url_val else "N/A"
                     scheme_dict["title"] = link_el.text.strip()
                     try: scheme_dict["department"] = card.find_element(By.CSS_SELECTOR, page_locators['dept']).text.strip()
                     except: scheme_dict["department"] = "N/A"
                     try: scheme_dict["overview"] = card.find_element(By.CSS_SELECTOR, page_locators['summary']).text.strip()
                     except: scheme_dict["overview"] = "N/A"
                     try:
                         tags = card.find_elements(By.CSS_SELECTOR, page_locators['category_tag'])
                         scheme_dict["keywords"] = ", ".join([t.text.strip() for t in tags if t.text.strip()]) or "N/A"
                     except: scheme_dict["keywords"] = "N/A"
                     # Add placeholders
                     scheme_dict["benefits"] = "Pending"; scheme_dict["eligibility"] = "Pending";
                     scheme_dict["process"] = "Pending"; scheme_dict["docs"] = "Pending";

                     if scheme_dict.get("title"): scraped_data_storage.append(scheme_dict)
                 except Exception as e_card: log.error("Error on card {} (Page 1): {}".format(i, e_card))

            log.info("Initial data collection yielded {} schemes.".format(len(scraped_data_storage)))
        except TimeoutException:
             log.critical("Could not load initial scheme cards. Aborting.")
             return

        # 4. Paginate and Collect More Summaries
        log.info("Beginning pagination process...")
        active_page = 1
        while len(scraped_data_storage) < SCHEMES_TO_COLLECT and active_page < MAX_PAGINATION_DEPTH:
            target_page = active_page + 1
            log.info("Navigating to page number {}...".format(target_page))
            xpath_page_link = page_locators['page_num_xpath'].format(target_page)
            try:
                page_target_link = web_driver_wait.until(EC.element_to_be_clickable((By.XPATH, xpath_page_link)))
                web_driver.execute_script("arguments[0].scrollIntoViewIfNeeded(true);", page_target_link)
                time.sleep(0.7)
                page_target_link.click()

                # Verify page change
                web_driver_wait.until(EC.text_to_be_present_in_element((By.XPATH, page_locators['current_page_xpath']), str(target_page)))
                log.info("Successfully loaded page {}.".format(target_page))
                active_page = target_page
                time.sleep(2.5) # Render pause

                # Scrape cards from current page
                current_cards = web_driver.find_elements(By.CSS_SELECTOR, page_locators['card'])
                log.info("Found {} cards on page {}.".format(len(current_cards), active_page))
                # (Extraction logic similar to page 1)
                for i, card in enumerate(current_cards):
                     if len(scraped_data_storage) >= SCHEMES_TO_COLLECT: break
                     scheme_dict = {}
                     try:
                         link_el = card.find_element(By.CSS_SELECTOR, page_locators['link'])
                         url_val = link_el.get_attribute('href')
                         scheme_dict["url"] = "{}/{}".format(BASE_URL.rstrip('/'), url_val.lstrip('/')) if url_val else "N/A"
                         scheme_dict["title"] = link_el.text.strip()
                         try: scheme_dict["department"] = card.find_element(By.CSS_SELECTOR, page_locators['dept']).text.strip()
                         except: scheme_dict["department"] = "N/A"
                         try: scheme_dict["overview"] = card.find_element(By.CSS_SELECTOR, page_locators['summary']).text.strip()
                         except: scheme_dict["overview"] = "N/A"
                         try:
                             tags = card.find_elements(By.CSS_SELECTOR, page_locators['category_tag'])
                             scheme_dict["keywords"] = ", ".join([t.text.strip() for t in tags if t.text.strip()]) or "N/A"
                         except: scheme_dict["keywords"] = "N/A"
                         scheme_dict["benefits"] = "Pending"; scheme_dict["eligibility"] = "Pending"; scheme_dict["process"] = "Pending"; scheme_dict["docs"] = "Pending";

                         if scheme_dict.get("title"): scraped_data_storage.append(scheme_dict)
                     except Exception as e_card: log.error("Error on card {} (Page {}): {}".format(i, active_page, e_card))

                log.info("Total schemes collected so far: {}.".format(len(scraped_data_storage)))
                if len(scraped_data_storage) >= SCHEMES_TO_COLLECT: log.info("Target count reached."); break

            except (TimeoutException, NoSuchElementException): log.warning("Could not find or navigate to page {}. Stopping pagination.".format(target_page)); break
            except Exception as e_page: log.error("Failed during pagination for page {}: {}".format(target_page, e_page)); break

        log.info("Summary collection phase complete. Total: {}.".format(len(scraped_data_storage)))

        # 5. Fetch Details for Each Scheme
        log.info("--- Starting Detail Extraction Phase ---")
        num_schemes = len(scraped_data_storage)
        for idx, scheme_record in enumerate(scraped_data_storage):
            detail_url = scheme_record.get("url")
            title = scheme_record.get("title", "Unknown Scheme")
            log.info("Processing details [{}/{}]: {}".format(idx + 1, num_schemes, title[:50]))

            if not detail_url or detail_url == "N/A":
                log.warning("Skipping due to missing URL for: {}".format(title))
                scheme_record.update({"benefits": "Invalid URL", "eligibility": "Invalid URL", "process": "Invalid URL", "docs": "Invalid URL"})
                continue

            try:
                web_driver.get(detail_url)
                web_driver_wait.until(EC.presence_of_element_located((By.XPATH, page_locators['detail_title_xpath'])))
                time.sleep(1) # Small render pause

                # Update record with details
                scheme_record["benefits"] = retrieve_detail_info(web_driver, page_locators['benefits_content_xpath'])
                scheme_record["eligibility"] = retrieve_detail_info(web_driver, page_locators['eligibility_content_xpath'])
                scheme_record["process"] = retrieve_detail_info(web_driver, page_locators['process_content_xpath'])
                scheme_record["docs"] = retrieve_detail_info(web_driver, page_locators['docs_content_xpath'])

            except TimeoutException:
                log.error("Timeout loading detail page for: {}".format(title))
                scheme_record.update({"benefits": "Load Timeout", "eligibility": "Load Timeout", "process": "Load Timeout", "docs": "Load Timeout"})
            except Exception as e_detail:
                log.exception("Error processing detail page for {}: {}".format(title, e_detail)) # Log full traceback
                scheme_record.update({"benefits": "Processing Error", "eligibility": "Processing Error", "process": "Processing Error", "docs": "Processing Error"})

            log.debug("Waiting before next detail request...")
            time.sleep(REQUEST_DELAY_SECONDS)

        log.info("Detail extraction phase complete.")

        # 6. Save Compiled Data
        log.info("--- Saving Collected Data ---")
        if scraped_data_storage:
            data_frame = pandas.DataFrame(scraped_data_storage)
            # Define desired column order
            final_columns = ["title", "department", "overview", "keywords", "benefits", "eligibility", "process", "docs", "url"]
            data_frame = data_frame.reindex(columns=final_columns) # Reorder/select columns
            # Save to CSV
            data_frame.to_csv(RESULT_FILENAME_CSV, index=False, encoding='utf-8-sig')
            log.info("Data successfully saved to {}".format(RESULT_FILENAME_CSV))
            # Save to JSON
            data_frame.to_json(RESULT_FILENAME_JSON, orient='records', indent=4, force_ascii=False)
            log.info("Data successfully saved to {}".format(RESULT_FILENAME_JSON))
            print("\nFinal data files created: {}, {}".format(RESULT_FILENAME_CSV, RESULT_FILENAME_JSON))
            print("\n--- Sample of Final Data (First 3 Rows) ---")
            print(data_frame.head(3)) # Basic print for sample
            print("--- ---")
        else:
            log.warning("No data was scraped. Output files not created.")
            print("No data collected, files not saved.")

    except Exception as e_global:
        # Catch any major exceptions not caught elsewhere
        log.critical("A major error occurred during the script execution: {}".format(e_global), exc_info=True)
        print("\nSCRIPT FAILED UNEXPECTEDLY.")

    finally:
        # 7. Cleanup: Ensure WebDriver is closed
        if web_driver is not None:
            log.info("Closing the WebDriver session.")
            web_driver.quit()
        else:
            log.info("WebDriver was not active or already closed.")

        end_ts = time.time()
        log.info("Scraper v2 finished. Total time: {:.2f} seconds.".format(end_ts - start_ts))
        print("\nScript execution completed.")

# Execute the main function when the script is run
if __name__ == "__main__":
    run_scraper()