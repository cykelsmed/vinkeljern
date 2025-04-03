"""
Formatters module for Vinkeljernet project.

This module provides functionality to format angle outputs in various formats
including JSON, Markdown, and HTML.
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from statistics import mean
from models import VinkelForslag


class BaseFormatter:
    """Base formatter interface for angle output."""
    
    def format_angles(self, angles: List[Dict[str, Any]], 
                      profile_name: Optional[str] = None,
                      topic: Optional[str] = None,
                      summary: Optional[str] = None) -> str:
        """
        Format a list of angles according to the formatter's output type.
        
        Args:
            angles: List of angle dictionaries
            profile_name: Name of the editorial profile used
            topic: The topic that angles were generated for
            summary: Optional summary text
            
        Returns:
            str: Formatted output
        """
        raise NotImplementedError("Subclasses must implement format_angles")
    
    def save_to_file(self, content: str, output_path: str) -> None:
        """
        Save formatted content to a file.
        
        Args:
            content: Formatted content to save
            output_path: Path to save the file to
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)


class JSONFormatter(BaseFormatter):
    """JSON formatter for angle output."""
    
    def format_angles(self, angles: List[Dict[str, Any]], 
                      profile_name: Optional[str] = None,
                      topic: Optional[str] = None,
                      summary: Optional[str] = None) -> str:
        """Format angles as JSON."""
        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "topic": topic or "Unknown topic",
                "profile": profile_name or "Unknown profile",
                "angle_count": len(angles)
            },
            "summary": summary or self._generate_summary(angles, topic),
            "angles": angles
        }
        
        return json.dumps(output, ensure_ascii=False, indent=2)
    
    def _generate_summary(self, angles: List[Dict[str, Any]], topic: Optional[str]) -> str:
        """Generate a summary of the angles."""
        if not angles:
            return "Ingen vinkler blev genereret."
            
        top_criteria = self._get_top_criteria(angles)
        return f"Genereret {len(angles)} vinkler om '{topic or 'emnet'}'. " + \
               f"De mest anvendte nyhedskriterier er: {', '.join(top_criteria)}."
    
    def _get_top_criteria(self, angles: List[Dict[str, Any]]) -> List[str]:
        """Get the most common news criteria across all angles."""
        criteria_count = {}
        for angle in angles:
            for criterion in angle.get("nyhedskriterier", []):
                criteria_count[criterion] = criteria_count.get(criterion, 0) + 1
                
        # Sort by count and get top 3
        top_criteria = sorted(criteria_count.items(), key=lambda x: x[1], reverse=True)[:3]
        return [c[0] for c in top_criteria]


class MarkdownFormatter(BaseFormatter):
    """Markdown formatter for angle output."""
    
    def format_angles(self, angles: List[Dict[str, Any]], 
                      profile_name: Optional[str] = None,
                      topic: Optional[str] = None,
                      summary: Optional[str] = None) -> str:
        """Format angles as Markdown."""
        now = datetime.now().strftime("%d-%m-%Y %H:%M")
        
        lines = [
            f"# Vinkelforslag til \"{topic or 'Emnet'}\"",
            f"_Genereret med Vinkeljernet {now} baseret på profilen \"{profile_name or 'Ikke angivet'}\")_",
            "",
            "## Sammenfatning",
            summary or self._generate_summary(angles, topic),
            "",
            "## Vinkler",
            ""
        ]
        
        # Add each angle
        for i, angle in enumerate(angles, 1):
            headline = angle.get('overskrift', 'Ingen overskrift')
            description = angle.get('beskrivelse', 'Ingen beskrivelse')
            rationale = angle.get('begrundelse', 'Ingen begrundelse')
            criteria = angle.get('nyhedskriterier', [])
            questions = angle.get('startSpørgsmål', [])
            score = angle.get('kriterieScore', None)
            background_info = angle.get('perplexityInfo', '') # Added perplexity info extraction
            
            lines.extend([
                f"### {i}. {headline}",
                "",
                description,
                "",
                f"**Begrundelse:** {rationale}",
                "",
                f"**Nyhedskriterier:** {', '.join(criteria)}",
                ""
            ])
            
            # Add perplexity background info if available
            if background_info and isinstance(background_info, str) and len(background_info.strip()) > 0:
                lines.append("**Baggrundsinformation:**")
                lines.append(background_info)
                lines.append("")
                
            # Add source suggestions if available
            source_info = angle.get('kildeForslagInfo', '')
            if source_info and isinstance(source_info, str) and len(source_info.strip()) > 0:
                lines.append("**Relevante kildeforslag:**")
                lines.append(source_info)
                lines.append("")
            
            if questions:
                lines.append("**Startspørgsmål:**")
                for q in questions:
                    lines.append(f"- {q}")
                lines.append("")
            
            if score is not None:
                lines.append(f"**Score:** {score}")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        # Add footer
        lines.extend([
            "## Om denne rapport",
            "Denne rapport er genereret automatisk med Vinkeljernet - et værktøj til at generere",
            "journalistiske vinkler baseret på redaktionelle DNA-profiler og kunstig intelligens.",
            "",
            f"_Rapport genereret: {now}_"
        ])
        
        return "\n".join(lines)
    
    def _generate_summary(self, angles: List[Dict[str, Any]], topic: Optional[str]) -> str:
        """Generate a summary of the angles in Markdown format."""
        if not angles:
            return "Ingen vinkler blev genereret."
            
        top_criteria = self._get_top_criteria(angles)
        avg_score = self._get_average_score(angles)
        
        summary = [
            f"Der er genereret **{len(angles)} vinkler** om '{topic or 'emnet'}'.",
            "",
            f"De mest anvendte nyhedskriterier er: **{', '.join(top_criteria)}**.",
        ]
        
        if avg_score:
            summary.append(f"Den gennemsnitlige score for vinklerne er **{avg_score:.1f}**.")
            
        return "\n".join(summary)
    
    def _get_top_criteria(self, angles: List[Dict[str, Any]]) -> List[str]:
        """Get the most common news criteria across all angles."""
        criteria_count = {}
        for angle in angles:
            for criterion in angle.get("nyhedskriterier", []):
                criteria_count[criterion] = criteria_count.get(criterion, 0) + 1
                
        # Sort by count and get top 3
        top_criteria = sorted(criteria_count.items(), key=lambda x: x[1], reverse=True)[:3]
        return [c[0] for c in top_criteria]
    
    def _get_average_score(self, angles: List[Dict[str, Any]]) -> Optional[float]:
        """Calculate the average score of all angles, if available."""
        scores = [a.get('kriterieScore', None) for a in angles]
        valid_scores = [s for s in scores if s is not None]
        
        if valid_scores:
            return mean(valid_scores)
        return None


class HTMLFormatter(BaseFormatter):
    """HTML formatter for angle output."""
    
    def format_angles(self, angles: List[Dict[str, Any]], 
                      profile_name: Optional[str] = None,
                      topic: Optional[str] = None,
                      summary: Optional[str] = None) -> str:
        """Format angles as HTML."""
        now = datetime.now().strftime("%d-%m-%Y %H:%M")
        
        # HTML head with embedded CSS
        html = [
            "<!DOCTYPE html>",
            "<html lang='da'>",
            "<head>",
            "    <meta charset='UTF-8'>",
            "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"    <title>Vinkelforslag: {topic or 'Emnet'}</title>",
            "    <style>",
            "        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }",
            "        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }",
            "        h2 { color: #2c3e50; margin-top: 30px; }",
            "        h3 { margin-top: 25px; color: #3498db; }",
            "        .angle { background-color: #f9f9f9; border-radius: 5px; padding: 15px; margin-bottom: 20px; border-left: 4px solid #3498db; }",
            "        .summary { background-color: #eef8ff; padding: 15px; border-radius: 5px; margin: 20px 0; }",
            "        .meta { color: #7f8c8d; font-size: 0.9em; }",
            "        .criteria { background-color: #e8f6fe; padding: 5px 10px; border-radius: 15px; display: inline-block; margin-right: 8px; font-size: 0.9em; }",
            "        .questions { padding-left: 20px; }",
            "        .questions li { margin-bottom: 8px; }",
            "        .rationale { font-style: italic; color: #555; }",
            "        .background-info { background-color: #f5f5f5; padding: 10px; border-left: 3px solid #3498db; margin: 10px 0; font-size: 0.9em; }",
            "        .source-info { background-color: #f5fffa; padding: 10px; border-left: 3px solid #2ecc71; margin: 10px 0; font-size: 0.9em; }",
            "        .score { font-weight: bold; color: #2980b9; }",
            "        .footer { margin-top: 40px; border-top: 1px solid #ddd; padding-top: 20px; color: #7f8c8d; font-size: 0.9em; }",
            "    </style>",
            "</head>",
            "<body>",
            f"    <h1>Vinkelforslag til \"{topic or 'Emnet'}\"</h1>",
            f"    <p class='meta'>Genereret med Vinkeljernet {now} baseret på profilen \"{profile_name or 'Ikke angivet'}\"</p>",
            "",
            "    <h2>Sammenfatning</h2>",
            f"    <div class='summary'>",
            f"        {summary or self._generate_summary(angles, topic)}",
            "    </div>",
            "",
            "    <h2>Vinkler</h2>"
        ]
        
        # Add each angle
        for i, angle in enumerate(angles, 1):
            headline = angle.get('overskrift', 'Ingen overskrift')
            description = angle.get('beskrivelse', 'Ingen beskrivelse')
            rationale = angle.get('begrundelse', 'Ingen begrundelse')
            criteria = angle.get('nyhedskriterier', [])
            questions = angle.get('startSpørgsmål', [])
            score = angle.get('kriterieScore', None)
            background_info = angle.get('perplexityInfo', '') # Added perplexity info extraction
            
            html.append(f"    <div class='angle'>")
            html.append(f"        <h3>{i}. {self._escape_html(headline)}</h3>")
            html.append(f"        <p>{self._escape_html(description)}</p>")
            html.append(f"        <p class='rationale'><strong>Begrundelse:</strong> {self._escape_html(rationale)}</p>")
            
            # Criteria as tags
            html.append("        <p><strong>Nyhedskriterier:</strong> ")
            for criterion in criteria:
                html.append(f"<span class='criteria'>{self._escape_html(criterion)}</span>")
            html.append("</p>")
            
            # Add perplexity background info if available
            if background_info and isinstance(background_info, str) and len(background_info.strip()) > 0:
                html.append(f"        <p><strong>Baggrundsinformation:</strong></p>")
                html.append(f"        <p class='background-info'>{self._escape_html(background_info)}</p>")
                
            # Add source suggestions if available
            source_info = angle.get('kildeForslagInfo', '')
            if source_info and isinstance(source_info, str) and len(source_info.strip()) > 0:
                html.append(f"        <p><strong>Relevante kildeforslag:</strong></p>")
                html.append(f"        <p class='source-info'>{self._escape_html(source_info)}</p>")
            
            # Questions as bullet points
            if questions:
                html.append("        <p><strong>Startspørgsmål:</strong></p>")
                html.append("        <ul class='questions'>")
                for q in questions:
                    html.append(f"            <li>{self._escape_html(q)}</li>")
                html.append("        </ul>")
            
            if score is not None:
                html.append(f"        <p><span class='score'>Score: {score}</span></p>")
            
            html.append("    </div>")
        
        # Add footer
        html.extend([
            "    <div class='footer'>",
            "        <p><strong>Om denne rapport</strong></p>",
            "        <p>Denne rapport er genereret automatisk med Vinkeljernet - et værktøj til at generere",
            "        journalistiske vinkler baseret på redaktionelle DNA-profiler og kunstig intelligens.</p>",
            f"        <p>Rapport genereret: {now}</p>",
            "    </div>",
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html)
    
    def _generate_summary(self, angles: List[Dict[str, Any]], topic: Optional[str]) -> str:
        """Generate a summary of the angles in HTML format."""
        if not angles:
            return "<p>Ingen vinkler blev genereret.</p>"
            
        top_criteria = self._get_top_criteria(angles)
        avg_score = self._get_average_score(angles)
        
        summary = [
            f"<p>Der er genereret <strong>{len(angles)} vinkler</strong> om '{topic or 'emnet'}'.</p>"
        ]
        
        summary.append("<p>De mest anvendte nyhedskriterier er: ")
        for criterion in top_criteria:
            summary.append(f"<span class='criteria'>{self._escape_html(criterion)}</span>")
        summary.append("</p>")
        
        if avg_score:
            summary.append(f"<p>Den gennemsnitlige score for vinklerne er <strong>{avg_score:.1f}</strong>.</p>")
            
        return "".join(summary)
    
    def _get_top_criteria(self, angles: List[Dict[str, Any]]) -> List[str]:
        """Get the most common news criteria across all angles."""
        criteria_count = {}
        for angle in angles:
            for criterion in angle.get("nyhedskriterier", []):
                criteria_count[criterion] = criteria_count.get(criterion, 0) + 1
                
        # Sort by count and get top 3
        top_criteria = sorted(criteria_count.items(), key=lambda x: x[1], reverse=True)[:3]
        return [c[0] for c in top_criteria]
    
    def _get_average_score(self, angles: List[Dict[str, Any]]) -> Optional[float]:
        """Calculate the average score of all angles, if available."""
        scores = [a.get('kriterieScore', None) for a in angles]
        valid_scores = [s for s in scores if s is not None]
        
        if valid_scores:
            return mean(valid_scores)
        return None
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def format_angles(angles: List[Dict[str, Any]], 
                  format_type: str = "markdown", 
                  profile_name: Optional[str] = None,
                  topic: Optional[str] = None,
                  summary: Optional[str] = None,
                  output_path: Optional[str] = None) -> str:
    """
    Format angles in the specified format.
    
    Args:
        angles: List of angle dictionaries
        format_type: Output format ("json", "markdown", or "html")
        profile_name: Name of the editorial profile used
        topic: Topic the angles were generated for
        summary: Optional custom summary
        output_path: Optional path to save the output to
        
    Returns:
        str: The formatted output
    """
    formatters = {
        "json": JSONFormatter(),
        "markdown": MarkdownFormatter(),
        "html": HTMLFormatter()
    }
    
    if format_type.lower() not in formatters:
        raise ValueError(f"Unsupported format: {format_type}. Must be one of: {', '.join(formatters.keys())}")
    
    formatter = formatters[format_type.lower()]
    output = formatter.format_angles(angles, profile_name, topic, summary)
    
    if output_path:
        formatter.save_to_file(output, output_path)
    
    return output