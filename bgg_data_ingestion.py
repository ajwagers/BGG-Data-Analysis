import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
import json
from typing import List, Dict, Any
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BGGDataIngester:
    def __init__(self, base_url: str = "https://boardgamegeek.com/xmlapi2"):
        self.base_url = base_url
        self.session = requests.Session()
        # Be respectful to BGG servers
        self.request_delay = 1.0
        
    def make_request(self, endpoint: str, params: Dict[str, Any] = None) -> ET.Element:
        """Make a request to BGG API with rate limiting and error handling."""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Check if we need to wait for processing (BGG returns 202 for queued requests)
            if response.status_code == 202:
                logger.info("Request queued, waiting for processing...")
                time.sleep(5)
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
        
        # BGG API allows up to 20 IDs per request
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
            "subdomain_ranks": {}
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

def main():
    """Example usage of the BGG data ingester."""
    ingester = BGGDataIngester()
    
    # Example 1: Get hot games and their details
    logger.info("Fetching hot games...")
    hot_games = ingester.get_hot_games()
    hot_game_ids = [game["id"] for game in hot_games]
    
    logger.info("Fetching detailed data for hot games...")
    detailed_games = ingester.get_game_details(hot_game_ids)
    ingester.save_to_files(detailed_games, "bgg_hot_games")
    
    # Example 2: Search for specific games
    logger.info("Searching for Wingspan...")
    search_results = ingester.search_games("Wingspan")
    if search_results:
        wingspan_details = ingester.get_game_details([search_results[0]["id"]])
        ingester.save_to_files(wingspan_details, "wingspan_data")
    
    # Example 3: Get top ranked games (you'd need to implement ranking logic)
    # For a portfolio project, you might want to get the top 1000 games
    logger.info("For a full dataset, you could fetch top-ranked games by ID range...")
    # top_game_ids = list(range(1, 1001))  # Example: first 1000 game IDs
    # Be careful with large requests - respect BGG's servers!
    
    logger.info("Data ingestion complete!")

if __name__ == "__main__":
    main()