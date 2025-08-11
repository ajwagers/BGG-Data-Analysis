import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
import json
from typing import List, Dict, Any
import logging
from pathlib import Path
from bs4 import BeautifulSoup
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BGGDataIngester:
    def __init__(self, base_url: str = "https://boardgamegeek.com/xmlapi2"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'BGG-Analysis-Script/1.0'})
        # Be respectful to BGG servers. A longer delay helps prevent 429 errors.
        self.request_delay = 2.5
        
    def make_request(self, endpoint: str, params: Dict[str, Any] = None) -> ET.Element:
        """Make a request to BGG API with rate limiting and error handling."""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params)

            # Handle 429 Too Many Requests by waiting and retrying recursively.
            if response.status_code == 429:
                wait_time = 30
                logger.warning(f"Received 429 Too Many Requests. Waiting {wait_time}s and retrying...")
                time.sleep(wait_time)
                return self.make_request(endpoint, params)

            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Check if we need to wait for processing (BGG returns 202 for queued requests)
            if response.status_code == 202:
                logger.info("Request queued by BGG, waiting 10 seconds before retrying...")
                time.sleep(10)
                return self.make_request(endpoint, params)
            
            # Rate limiting
            time.sleep(self.request_delay)
            return root
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
        except ET.ParseError as e:
            logger.error(f"XML parsing failed: {e}")
            raise
    
    def get_game_details(self, game_ids: List[int]) -> List[Dict[str, Any]]:
        """Fetch detailed information for a list of game IDs."""
        games_data = []
        
        # BGG API can be slow or return 202 (Accepted) for very large batches.
        # While the API can handle multiple IDs, very large requests can fail.
        # A smaller, more conservative batch size is more reliable.
        batch_size = 20
        
        for i in range(0, len(game_ids), batch_size):
            batch_ids = game_ids[i:i + batch_size]
            ids_str = ",".join(map(str, batch_ids))
            
            logger.info(f"Fetching games {i+1}-{min(i+batch_size, len(game_ids))} of {len(game_ids)}")
            
            root = self.make_request("thing", {"id": ids_str, "stats": 1})
            
            for item in root.findall("item"):
                game_data = self._parse_game_item(item)
                games_data.append(game_data)
        
        return games_data
    
    def _parse_game_item(self, item: ET.Element) -> Dict[str, Any]:
        """Parse a single game item from BGG XML."""
        game_data = {
            "id": int(item.get("id")),
            "type": item.get("type"),
            "thumbnail": "",
            "image": "",
            "names": [],
            "description": "",
            "year_published": None,
            "min_players": None,
            "max_players": None,
            "playing_time": None,
            "min_playing_time": None,
            "max_playing_time": None,
            "min_age": None,
            "categories": [],
            "mechanics": [],
            "families": [],
            "designers": [],
            "artists": [],
            "publishers": [],
            "expansions": [],
            "implements": [],
            "users_rated": None,
            "average_rating": None,
            "bayes_average": None,
            "stddev": None,
            "median": None,
            "owned": None,
            "trading": None,
            "wanting": None,
            "wishing": None,
            "num_comments": None,
            "num_weights": None,
            "average_weight": None,
            "bgg_rank": None,
            "subdomain_ranks": {},
            **{f"votes_{i}": 0 for i in range(1, 11)}
        }
        
        # Basic info
        thumbnail = item.find("thumbnail")
        if thumbnail is not None:
            game_data["thumbnail"] = thumbnail.text or ""
            
        image = item.find("image")
        if image is not None:
            game_data["image"] = image.text or ""
        
        # Names
        for name in item.findall("name"):
            name_data = {
                "type": name.get("type"),
                "sort_index": int(name.get("sortindex", 1)),
                "value": name.get("value")
            }
            game_data["names"].append(name_data)
        
        # Description
        description = item.find("description")
        if description is not None:
            game_data["description"] = description.text or ""
        
        # Year published
        year = item.find("yearpublished")
        if year is not None:
            game_data["year_published"] = int(year.get("value", 0)) or None
        
        # Player counts and times
        for attr in ["minplayers", "maxplayers", "playingtime", "minplaytime", "maxplaytime", "minage"]:
            element = item.find(attr)
            if element is not None:
                key = attr.replace("players", "_players").replace("time", "_playing_time").replace("minage", "min_age")
                game_data[key] = int(element.get("value", 0)) or None
        
        # Categories, mechanics, families, etc.
        link_mappings = {
            "boardgamecategory": "categories",
            "boardgamemechanic": "mechanics",
            "boardgamefamily": "families",
            "boardgamedesigner": "designers",
            "boardgameartist": "artists",
            "boardgamepublisher": "publishers",
            "boardgameexpansion": "expansions",
            "boardgameimplementation": "implements"
        }
        
        for link in item.findall("link"):
            link_type = link.get("type")
            if link_type in link_mappings:
                game_data[link_mappings[link_type]].append({
                    "id": int(link.get("id")),
                    "value": link.get("value")
                })
        
        # Statistics
        statistics = item.find("statistics")
        if statistics is not None:
            ratings = statistics.find("ratings")
            if ratings is not None:
                # Basic ratings
                for stat in ["usersrated", "average", "bayesaverage", "stddev", "median", "owned", "trading", "wanting", "wishing", "numcomments", "numweights"]:
                    element = ratings.find(stat)
                    if element is not None:
                        key = stat.replace("users", "users_").replace("bayes", "bayes_").replace("num", "num_")
                        value = element.get("value")
                        if value and value != "0":
                            game_data[key] = float(value) if stat in ["average", "bayesaverage", "stddev", "median"] else int(value)
                
                # Average weight
                avgweight = ratings.find("averageweight")
                if avgweight is not None:
                    value = avgweight.get("value")
                    if value and value != "0":
                        game_data["average_weight"] = float(value)
                
                # Rankings
                ranks = ratings.find("ranks")
                if ranks is not None:
                    for rank in ranks.findall("rank"):
                        rank_type = rank.get("type")
                        rank_name = rank.get("name")
                        rank_value = rank.get("value")
                        
                        if rank_value and rank_value.isdigit():
                            if rank_name == "boardgame":
                                game_data["bgg_rank"] = int(rank_value)
                            else:
                                game_data["subdomain_ranks"][rank_name] = int(rank_value)
        
        # Add parsing for rating distribution poll
        for poll in item.findall("poll"):
            if poll.get("name") == "boardgame_rating":
                results = poll.find("results")
                if results is not None:
                    for result in results.findall("result"):
                        level = result.get("level")
                        numvotes = result.get("numvotes")
                        if level and numvotes and level.isdigit():
                            key = f"votes_{level}"
                            if key in game_data:
                                game_data[key] = int(numvotes)
        
        return game_data
    
    def get_hot_games(self, game_type: str = "boardgame") -> List[Dict[str, Any]]:
        """Get the current hot games list."""
        root = self.make_request("hot", {"type": game_type})
        
        hot_games = []
        for item in root.findall("item"):
            hot_games.append({
                "id": int(item.get("id")),
                "rank": int(item.get("rank")),
                "thumbnail": item.find("thumbnail").get("value") if item.find("thumbnail") is not None else "",
                "name": item.find("name").get("value") if item.find("name") is not None else "",
                "year_published": int(item.find("yearpublished").get("value")) if item.find("yearpublished") is not None else None
            })
        
        return hot_games
    
    def search_games(self, query: str, exact: bool = False) -> List[Dict[str, Any]]:
        """Search for games by name."""
        params = {"query": query, "type": "boardgame"}
        if exact:
            params["exact"] = 1
            
        root = self.make_request("search", params)
        
        results = []
        for item in root.findall("item"):
            results.append({
                "id": int(item.get("id")),
                "type": item.get("type"),
                "name": item.find("name").get("value") if item.find("name") is not None else "",
                "year_published": int(item.find("yearpublished").get("value")) if item.find("yearpublished") is not None else None
            })
        
        return results
    
    def save_to_files(self, data: List[Dict[str, Any]], base_filename: str):
        """Save data to both JSON and CSV formats."""
        # Save as JSON (preserves complex data structures)
        json_file = f"{base_filename}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} records to {json_file}")
        
        # Flatten data for CSV
        flattened_data = []
        for game in data:
            flat_game = game.copy()
            
            # Convert lists to comma-separated strings for CSV
            for key in ["categories", "mechanics", "families", "designers", "artists", "publishers"]:
                if flat_game[key]:
                    flat_game[key] = "; ".join([item["value"] for item in flat_game[key]])
                else:
                    flat_game[key] = ""
            
            # Handle names (take primary name)
            primary_name = ""
            if flat_game["names"]:
                primary_names = [n for n in flat_game["names"] if n["type"] == "primary"]
                if primary_names:
                    primary_name = primary_names[0]["value"]
                elif flat_game["names"]:
                    primary_name = flat_game["names"][0]["value"]
            flat_game["primary_name"] = primary_name
            del flat_game["names"]
            
            # Remove complex nested data that doesn't work well in CSV
            for key in ["expansions", "implements", "subdomain_ranks"]:
                del flat_game[key]
            
            flattened_data.append(flat_game)
        
        # Save as CSV
        df = pd.DataFrame(flattened_data)
        csv_file = f"{base_filename}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"Saved flattened data to {csv_file}")

def load_ids_from_csv(filepath: str, id_column: str = 'id') -> List[int]:
    """Loads a list of game IDs from a CSV file."""
    logger.info(f"Loading base game IDs from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        if id_column not in df.columns:
            logger.error(f"ID column '{id_column}' not found in {filepath}.")
            return []
        
        # Drop rows with missing IDs and convert to int
        ids = df[id_column].dropna().astype(int).tolist()
        logger.info(f"Loaded {len(ids)} IDs from CSV.")
        return ids
    except FileNotFoundError:
        logger.error(f"Archive CSV file not found: {filepath}. Starting with an empty list.")
        return []
    except Exception as e:
        logger.error(f"Error loading CSV {filepath}: {e}")
        return []

def scrape_top_ids_by_year(session: requests.Session, start_year: int, pages_to_scrape: int = 20) -> List[int]:
    """
    Scrapes BGG advanced search for top-ranked games published since a given year.
    """
    game_ids = []
    consecutive_failures = 0
    logger.info(f"Scraping {pages_to_scrape} pages for top games published since {start_year}...")
    
    for page in range(1, pages_to_scrape + 1):
        # This URL uses the advanced search, sorted by rank, filtered by year published.
        browse_url = (
            f"https://boardgamegeek.com/search/boardgame/page/{page}"
            f"?sort=rank&advsearch=1&range%5Byearpublished%5D%5Bmin%5D={start_year}"
        )
        try:
            response = session.get(browse_url)
            response.raise_for_status()
            
            initial_id_count = len(game_ids)
            soup = BeautifulSoup(response.content, 'lxml')
            
            # The selector for advanced search results is different from the simple browse page.
            links = soup.select('td.collection_objectname > div > a')
            if not links:
                logger.warning(f"No game links found on page {page}. The page structure may have changed.")
                consecutive_failures += 1
            else:
                consecutive_failures = 0

            for link in links:
                href = link.get('href')
                if href and '/boardgame/' in href:
                    try:
                        game_id = int(href.split('/')[2])
                        if game_id not in game_ids:
                            game_ids.append(game_id)
                    except (ValueError, IndexError):
                        continue
            
            if len(game_ids) == initial_id_count and links:
                logger.warning(f"No new unique game IDs found on page {page}.")

            logger.info(f"Scraped page {page}/{pages_to_scrape}, found {len(game_ids)} total unique IDs so far.")

            if consecutive_failures >= 3:
                logger.error("Failed to find game links for 3 consecutive pages. Aborting scrape.")
                break

            time.sleep(0.5) # Be respectful
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to scrape page {page}: {e}")
            break
            
    return game_ids

def main():
    """Example usage of the BGG data ingester."""
    ingester = BGGDataIngester()
    
    # --- Configuration ---
    # Assumes you have a CSV named 'bgg_top_games_archive.csv' in the same directory.
    # This file should contain an 'id' column with BGG game IDs.
    archive_csv_path = 'initial_data\\games.csv'
    output_filename = 'bgg_top_games_updated'
    current_year = datetime.datetime.now().year
    start_year_for_scraping = current_year - 4
    pages_of_new_games_to_scrape = 20 # Scrape ~2000 recent top games
    
    # 1. Load game IDs from the archive CSV.
    base_game_ids = load_ids_from_csv(archive_csv_path, id_column='BGGId')
    
    # 2. Scrape for top-ranked games from recent years to fill in the gaps.
    with requests.Session() as scrape_session:
        scrape_session.headers.update({'User-Agent': 'BGG-Analysis-Script/1.0'})
        recent_game_ids = scrape_top_ids_by_year(
            scrape_session, 
            start_year=start_year_for_scraping,
            pages_to_scrape=pages_of_new_games_to_scrape
        )
        
    # 3. Combine the lists and remove duplicates.
    combined_ids = list(set(base_game_ids + recent_game_ids))
    logger.info(f"Combined list contains {len(combined_ids)} unique game IDs.")
    
    # 4. Use the API to get detailed data for the combined list.
    if combined_ids:
        logger.info(f"Fetching detailed data for {len(combined_ids)} games...")
        detailed_games = ingester.get_game_details(combined_ids)

        # Filter out games that didn't return data or have no rank
        ranked_games = [g for g in detailed_games if g.get('bgg_rank') is not None]
        logger.info(f"Found {len(ranked_games)} games with rank information.")

        # Sort final list by BGG rank (lower is better)
        ranked_games.sort(key=lambda x: x['bgg_rank'])

        # 5. Save the results
        ingester.save_to_files(ranked_games, output_filename)

    logger.info("Data ingestion complete!")

if __name__ == "__main__":
    main()