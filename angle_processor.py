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
        if not hits_nogo_areas(angle, profile.nogo_områder):
            # Calculate a score based on news criteria priority weights
            angle["calculatedScore"] = calculate_angle_score(angle, profile.nyhedsprioritering)
            filtered_angles.append(angle)
    
    # Sort angles by score (descending)
    ranked_angles = sorted(filtered_angles, key=lambda x: x["calculatedScore"], reverse=True)
    
    # Ensure diversity in the selection
    diverse_angles = ensure_diverse_angles(ranked_angles, num_angles)
    
    return diverse_angles

def hits_nogo_areas(angle: Dict[str, Any], nogo_areas: List[str]) -> bool:
    """
    Check if an angle hits any no-go areas.
    
    Args:
        angle: The angle object
        nogo_areas: List of no-go areas
        
    Returns:
        bool: True if the angle hits a no-go area, False otherwise
    """
    angle_text = (angle["overskrift"] + " " + angle["beskrivelse"]).lower()
    
    for nogo in nogo_areas:
        # Simple keyword matching - could be improved with more sophisticated NLP
        if any(term.lower() in angle_text for term in nogo.split()):
            return True
    
    return False

def calculate_angle_score(angle: Dict[str, Any], priorities: Dict[str, int]) -> float:
    """
    Calculate a score for an angle based on the profile's news criteria priorities.
    
    Args:
        angle: The angle object
        priorities: Dictionary of news criteria and their weights
        
    Returns:
        float: Calculated score
    """
    # If angle already has a kriterieScore, use that as a starting point
    score = angle.get("kriterieScore", 0)
    
    # Add weights from the priorities
    for criterion in angle.get("nyhedskriterier", []):
        if criterion in priorities:
            score += priorities[criterion]
    
    return score

def ensure_diverse_angles(ranked_angles: List[Dict[str, Any]], num_angles: int) -> List[Dict[str, Any]]:
    """
    Ensure diversity in the selected angles by avoiding similar topics.
    
    Args:
        ranked_angles: Sorted list of angles by score
        num_angles: Number of angles to select
        
    Returns:
        List[Dict]: Diverse selection of angles
    """
    if len(ranked_angles) <= num_angles:
        return ranked_angles
    
    diverse_selection = [ranked_angles[0]]  # Start with the highest-scoring angle
    remaining = ranked_angles[1:]
    
    # Simple approach: greedily select angles that are most different from already selected ones
    while len(diverse_selection) < num_angles and remaining:
        most_diverse_index = find_most_diverse_angle(diverse_selection, remaining)
        diverse_selection.append(remaining.pop(most_diverse_index))
    
    return diverse_selection

def find_most_diverse_angle(selected: List[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> int:
    """
    Find the index of the most diverse angle from the candidates compared to already selected angles.
    
    Args:
        selected: Already selected angles
        candidates: Candidate angles to choose from
        
    Returns:
        int: Index of the most diverse candidate
    """
    # Simple approach using word overlap as a diversity measure
    best_index = 0
    lowest_overlap = float('inf')
    
    for i, candidate in enumerate(candidates):
        # Create a bag of words for the candidate
        candidate_text = (candidate["overskrift"] + " " + candidate["beskrivelse"]).lower().split()
        candidate_words = set(candidate_text)
        
        # Calculate overlap with all selected angles
        total_overlap = 0
        for sel in selected:
            sel_text = (sel["overskrift"] + " " + sel["beskrivelse"]).lower().split()
            sel_words = set(sel_text)
            
            # Calculate Jaccard similarity (intersection over union)
            intersection = len(candidate_words.intersection(sel_words))
            union = len(candidate_words.union(sel_words))
            if union > 0:
                similarity = intersection / union
                total_overlap += similarity
        
        # Keep track of the candidate with lowest overlap
        if total_overlap < lowest_overlap:
            lowest_overlap = total_overlap
            best_index = i
    
    return best_index