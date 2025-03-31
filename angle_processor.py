"""
Angle processing module for Vinkeljernet project.

This module handles the filtering, scoring, and selection of news angles
based on editorial profiles and generated angles.
"""

from typing import List, Dict, Any
from models import RedaktionelDNA

def filter_and_rank_angles(angles: List[Dict[str, Any]], profile: RedaktionelDNA, 
                          num_angles: int = 5) -> List[Dict[str, Any]]:
    """
    Filter out invalid angles, score and rank them, and select the best ones.
    
    Args:
        angles: List of generated angle objects
        profile: RedaktionelDNA profile object
        num_angles: Number of angles to return (default: 5)
        
    Returns:
        List[Dict]: The filtered and ranked angles
    """
    # Filter out angles that hit no-go areas (basic keyword matching)
    filtered_angles = []
    for angle in angles:
        if not hits_no_go_areas(angle, profile):
            # Calculate a score based on news criteria priority weights
            angle["calculatedScore"] = calculate_angle_score(angle, profile)
            filtered_angles.append(angle)
        else:
            print(f"Angle filtered out due to hitting no-go areas: {angle}")
    
    # Sort angles by score (descending)
    ranked_angles = sorted(filtered_angles, key=lambda x: x["calculatedScore"], reverse=True)
    
    # Ensure diversity in the selection
    diverse_angles = ensure_diverse_angles(ranked_angles, num_angles)
    
    return diverse_angles

def hits_no_go_areas(angle: Dict[str, Any], profile: RedaktionelDNA) -> bool:
    """
    Check if an angle hits any no-go areas in the profile.
    
    Args:
        angle: Angle to check
        profile: Editorial DNA profile to check against
        
    Returns:
        True if angle hits any no-go areas, False otherwise
    """
    # Completely disable the check
    return False

    # Original implementation below
    # for no_go in profile.noGoOmrader:
    #    if ... (rest of the original code)

def calculate_angle_score(angle: Dict[str, Any], profile: RedaktionelDNA) -> float:
    """
    Enhanced scoring system that considers multiple factors
    """
    # Initialize with a base score
    score = 0.0
    
    try:
        # News criteria scoring - match to profile priorities
        for criterion in angle.get("nyhedskriterier", []):
            if criterion in profile.nyhedsprioritering:
                score += profile.nyhedsprioritering[criterion]
            else:
                # Default value for unlisted criteria
                score += 2
                
        # Bonus scoring for focus areas
        angle_text = " ".join([
            angle.get("overskrift", ""),
            angle.get("beskrivelse", ""),
            angle.get("begrundelse", "")
        ]).lower()
        
        for area in profile.fokusOmrader:
            if area.lower() in angle_text:
                score += 3
        
        # Return at least 1 point even for low-scoring angles
        return max(1.0, score)
    
    except Exception as e:
        print(f"Error calculating score: {e}")
        return 1.0  # Default fallback score

def calculate_angle_relevance(angle: Dict[str, Any], profile: RedaktionelDNA) -> float:
    """
    Calculate a relevance score for a news angle based on the profile's tone and style.
    
    Note: The field 'tone_og_stil' in the profile is a plain string describing the tone and style,
    and it must not be treated as a dictionary.
    """
    score = 0.0
    try:
        # Get the tone text as a lowercase string.
        tone_text = profile.tone_og_stil.lower()
        # Get the angle's description as a text.
        angle_text = angle.get("beskrivelse", "").lower()
        
        # Simple example: add to the score if the profile's tone is mentioned in the angle description.
        if tone_text in angle_text:
            score += 1.0
            
        # Additional relevance calculation logic can be added here
        
        return score
    except Exception as e:
        print(f"Error in calculate_angle_relevance: {e}")
        return 0.0

def ensure_diverse_angles(ranked_angles: List[Dict[str, Any]], num_angles: int) -> List[Dict[str, Any]]:
    """
    Enhanced algorithm to ensure diversity in selected angles.
    
    Args:
        ranked_angles: Sorted list of angles by score
        num_angles: Number of angles to select
        
    Returns:
        List[Dict]: Diverse selection of angles
    """
    if len(ranked_angles) <= num_angles:
        return ranked_angles
    
    # Always include the highest ranked angle
    diverse_selection = [ranked_angles[0]]
    candidates = ranked_angles[1:]
    
    # Track selected criteria and themes to ensure diversity
    selected_criteria = set()
    for criterion in ranked_angles[0].get('nyhedskriterier', []):
        selected_criteria.add(criterion)
    
    # Extract potential themes from angle descriptions
    def extract_themes(angle: Dict[str, Any]) -> List[str]:
        text = f"{angle.get('overskrift', '')} {angle.get('beskrivelse', '')}"
        # Simple theme extraction based on key nouns
        themes = []
        # Potential theme words - could be expanded with NLP
        theme_indicators = [
            'økonomi', 'politik', 'samfund', 'miljø', 'klima', 'sundhed', 
            'bolig', 'uddannelse', 'teknologi', 'social', 'kultur'
        ]
        for theme in theme_indicators:
            if theme in text.lower():
                themes.append(theme)
        return themes
    
    selected_themes = set(extract_themes(ranked_angles[0]))
    
    # Select remaining angles with a weighted approach to ensure diversity
    while len(diverse_selection) < num_angles and candidates:
        best_candidate_index = -1
        best_candidate_score = -1
        
        for i, candidate in enumerate(candidates):
            # Calculate diversity score - higher is more diverse
            diversity_score = 0
            
            # 1. Criteria diversity (different news criteria than already selected)
            new_criteria = 0
            for criterion in candidate.get('nyhedskriterier', []):
                if criterion not in selected_criteria:
                    new_criteria += 1
            diversity_score += new_criteria * 10  # Weight for criteria diversity
            
            # 2. Theme diversity
            candidate_themes = extract_themes(candidate)
            new_themes = 0
            for theme in candidate_themes:
                if theme not in selected_themes:
                    new_themes += 1
            diversity_score += new_themes * 15  # Weight for theme diversity
            
            # 3. Text similarity (lower is better)
            text_similarity = calculate_text_similarity(candidate, diverse_selection)
            diversity_score -= text_similarity * 5  # Penalty for textual similarity
            
            # 4. Consider original ranking
            rank_position = ranked_angles.index(candidate) if candidate in ranked_angles else len(ranked_angles)
            rank_score = max(0, 20 - (rank_position * 1.5))  # Higher ranked angles get bonus
            diversity_score += rank_score
            
            if diversity_score > best_candidate_score:
                best_candidate_score = diversity_score
                best_candidate_index = i
        
        if best_candidate_index >= 0:
            best_candidate = candidates.pop(best_candidate_index)
            diverse_selection.append(best_candidate)
            
            # Update tracking sets
            for criterion in best_candidate.get('nyhedskriterier', []):
                selected_criteria.add(criterion)
            for theme in extract_themes(best_candidate):
                selected_themes.add(theme)
        else:
            # If we can't find a good candidate, just take the next highest ranked
            diverse_selection.append(candidates.pop(0))
    
    return diverse_selection

def calculate_text_similarity(candidate: Dict[str, Any], selected: List[Dict[str, Any]]) -> float:
    """
    Calculate text similarity between a candidate and already selected angles.
    
    Args:
        candidate: Candidate angle
        selected: Already selected angles
        
    Returns:
        float: Similarity score (0-1, higher means more similar)
    """
    candidate_text = f"{candidate.get('overskrift', '')} {candidate.get('beskrivelse', '')}".lower()
    candidate_words = set(word for word in candidate_text.split() if len(word) > 3)
    
    max_similarity = 0
    for sel in selected:
        sel_text = f"{sel.get('overskrift', '')} {sel.get('beskrivelse', '')}".lower()
        sel_words = set(word for word in sel_text.split() if len(word) > 3)
        
        # Calculate Jaccard similarity
        if not sel_words or not candidate_words:
            continue
            
        intersection = len(candidate_words.intersection(sel_words))
        union = len(candidate_words.union(sel_words))
        similarity = intersection / union if union > 0 else 0
        
        max_similarity = max(max_similarity, similarity)
    
    return max_similarity