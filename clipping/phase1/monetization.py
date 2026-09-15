"""
YouTube Shorts Monetization Optimizer

Integrates research findings on:
- VVSA (Viewed vs Swiped Away) optimization
- Retention curve optimization
- Hook strength detection
- Length & structure optimization
- Series detection for binge-ability

All components designed to be independently testable.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MonetizationMetrics:
    """Monetization prediction for a clip."""
    hook_strength: float  # 0-1: How strong is the opening hook?
    vvsa_score: float  # 0-1: Likelihood of not swiping away (first 1-3s)
    retention_score: float  # 0-1: Predicted avg view duration %
    optimal_length: float  # seconds
    structure_score: float  # 0-1: Does it follow retention curve template?
    monetization_score: float  # 0-1: Overall monetization potential
    series_potential: str  # "high", "medium", "low", "not_applicable"
    recommendations: List[str]
    has_face: Optional[bool] = None  # None = unknown (no detector run)
    has_motion: Optional[bool] = None  # None = unknown (no detector run)
    value_density_score: float = 0.5  # 0-1: dense/tight vs. filler-laden
    loop_score: float = 0.5  # 0-1: does the ending flow back into the hook?


class HookStrengthDetector:
    """Detects opening hook strength (critical for VVSA)."""

    STRONG_OPENING_PHRASES = [
        # Tech/Dev specific
        'stop', 'never', 'this one', 'don\'t', 'mistake', 'wrong', 'biggest',
        'secret', 'hack', 'trick', 'actually', 'nobody knows', 'most people',
        'i was wrong', 'that\'s not', 'forget', 'you\'re doing it wrong',
        # Question format
        'why', 'how', 'what if', 'can you', 'would you', 'should you',
        # Urgency/stakes
        'fast', 'quick', 'easy', 'simple', 'before', 'after', 'save',
        'lose', 'crash', 'break', 'warning', 'danger', 'crazy', 'insane',
        # Curiosity gap ("nobody talks about this")
        'nobody talks about', 'hidden', 'quietly', 'no one tells you',
        'the truth about', 'what they don\'t tell you',
        # Bold claim / contrarian
        'you don\'t need', 'stop learning', 'stop doing this', 'is a lie',
        'is wasting your time', 'is not', 'don\'t do this',
        # Direct call-out ("if you're a backend dev...")
        'if you\'re', 'if you build', 'if you code', 'if you use',
        'if you live', 'if you love', 'if you\'re planning',
        # Result-first / blueprint
        'here\'s the exact', 'here\'s how', 'here\'s the setup',
        'here\'s what happened', 'this is the exact', 'this is how i',
        # Specific numbers / stakes
        'i tested', 'i wasted', 'i spent', 'i tried', 'i made this',
        'this cut my', 'this reduced my', 'this saved me',
    ]

    # Anti-patterns (weak openings)
    WEAK_OPENING_PHRASES = [
        'hey', 'hi', 'welcome', 'thanks for', 'don\'t forget to', 'like and subscribe',
        'so', 'basically', 'okay', 'alright', 'you know', 'i mean', 'uh', 'um',
        'let me', 'now let\'s', 'first let\'s', 'as you can see'
    ]

    @classmethod
    def detect_hook_strength(cls, opening_text: str, is_question: bool = False) -> Tuple[float, str]:
        """
        Detect hook strength of opening 1-3 seconds of clip.

        Args:
            opening_text: First sentence/phrase of clip transcript
            is_question: Is it a question?

        Returns:
            (score 0-1, reason)
        """
        if not opening_text or len(opening_text) < 3:
            return 0.0, "No opening text"

        score = 0.5  # Base score
        reasons = []
        text_lower = opening_text.lower()

        # Question bonus
        if is_question or '?' in opening_text:
            score += 0.15
            reasons.append("Question opening (+15%)")

        # Strong opening words
        for phrase in cls.STRONG_OPENING_PHRASES:
            if phrase in text_lower:
                score += 0.2
                reasons.append(f"Strong phrase: '{phrase}' (+20%)")
                break  # Only count strongest match

        # Weak opening words penalty
        for phrase in cls.WEAK_OPENING_PHRASES:
            if text_lower.startswith(phrase) or f" {phrase} " in f" {text_lower} ":
                score -= 0.15
                reasons.append(f"Weak opening: '{phrase}' (-15%)")
                break

        # Exclamation (emphasis)
        if opening_text.endswith('!') or opening_text.count('!') > 0:
            score += 0.1
            reasons.append("Exclamation (+10%)")

        # Length (ideal 8-15 words for hook)
        word_count = len(opening_text.split())
        if 8 <= word_count <= 15:
            score += 0.1
            reasons.append(f"Optimal length {word_count}w (+10%)")
        elif word_count > 25:
            score -= 0.1
            reasons.append(f"Too long {word_count}w (-10%)")

        final_score = min(1.0, max(0.0, score))
        reason = " | ".join(reasons) if reasons else "Neutral opening"

        return final_score, reason

    @classmethod
    def batch_detect(cls, segments: List[Dict]) -> List[Dict]:
        """Detect hook strength for multiple segments."""
        for segment in segments:
            text = segment.get('text', '')
            hook_score, reason = cls.detect_hook_strength(text)
            segment['hook_score'] = hook_score
            segment['hook_reason'] = reason
        return segments


class VVSAPredictor:
    """Predicts VVSA (Viewed vs Swiped Away) resistance for first 1-3 seconds."""

    # Factors that reduce swipe-away
    MOTION_INDICATORS = ['cut', 'switch', 'flashes', 'zoom', 'pan', 'shakes', 'fast', 'quick']
    FACE_PRESENCE_VALUE = 0.15  # Having a face in first 3s
    SHOCK_FACTOR_PHRASES = [
        'wait', 'hold on', 'but here\'s', 'actually', 'watch this', 'look',
        # Emotional / share triggers (surprise, awe, FOMO, save-worthiness)
        'shocking', 'unbelievable', 'you won\'t believe', 'this changed everything',
        'save this', 'save this for later', 'i can\'t believe', 'mind blown',
        'this is insane', 'this doesn\'t feel real', 'plot twist',
    ]

    @classmethod
    def predict_vvsa(
        cls,
        opening_text: str,
        has_motion: Optional[bool] = None,
        has_face: Optional[bool] = None,
    ) -> float:
        """
        Predict likelihood of viewer NOT swiping away in first 1-3s.

        Target: >70%, ideally 80%+

        Args:
            opening_text: Opening text of clip
            has_motion: Does video have motion/cuts in first 3s?
                None = unknown (no detector run) -> neutral contribution.
            has_face: Is there a face visible in first 3s?
                None = unknown (no detector run) -> neutral contribution.

        Returns:
            Score 0-1 (0 = high swipe-away, 1 = very stickily)
        """
        score = 0.5  # Base

        # Hook strength contributes heavily to VVSA
        hook_score, _ = HookStrengthDetector.detect_hook_strength(opening_text)
        score += hook_score * 0.3  # Hook is 30% of VVSA

        # Motion in first 3s (None = unknown, no data -> no adjustment)
        if has_motion is True:
            score += 0.2
        elif has_motion is False:
            score -= 0.1

        # Face presence (None = unknown, no data -> no adjustment)
        if has_face is True:
            score += cls.FACE_PRESENCE_VALUE
        elif has_face is False:
            score -= 0.05

        # Shock/curiosity factor
        text_lower = opening_text.lower()
        for phrase in cls.SHOCK_FACTOR_PHRASES:
            if phrase in text_lower:
                score += 0.1
                break

        # Punctuation (shows emphasis)
        punctuation_count = opening_text.count('!') + opening_text.count('?')
        if punctuation_count > 0:
            score += 0.05 * min(2, punctuation_count)

        return min(1.0, max(0.0, score))


class RetentionCurveOptimizer:
    """Optimizes clip structure for retention (avg view duration)."""

    OPTIMAL_STRUCTURE = {
        'hook': {'start': 0, 'end': 3, 'target_text': 'Strong opening promise'},
        'stakes': {'start': 3, 'end': 10, 'target_text': 'Why this matters'},
        'payoff': {'start': 10, 'end': 30, 'target_text': 'Core insight/demo'},
        'cta': {'start': 30, 'end': 35, 'target_text': 'Quick CTA or hook back'}
    }

    @classmethod
    def predict_retention(cls, clip_duration: float, has_good_hook: bool = False,
                         has_payoff: bool = False, has_cta: bool = False) -> float:
        """
        Predict average view duration percentage.

        Target: 73-76%+ for viral Shorts
        Formula: Based on structure + length

        Returns:
            Score 0-1 (percentage of clip watched)
        """
        # Length optimization.
        # Research on Shorts completion rates favors a wider 15-45s sweet
        # spot than pure ad-inventory targets: tight sub-25s clips often
        # loop and can exceed 100% completion, so they're rewarded almost
        # as much as the 20-35s core, not penalized the way the old
        # 25-40s-only curve did.
        score = 0.5  # Base

        if 20 <= clip_duration <= 35:
            score += 0.22  # Ideal length (dense, loop-friendly)
        elif 15 <= clip_duration < 20 or 35 < clip_duration <= 45:
            score += 0.12  # Still strong
        elif clip_duration < 15:
            score -= 0.05  # Very short - can still loop, small penalty
        elif clip_duration > 55:
            score -= 0.2  # Too long (swipe-away risk)
        elif clip_duration > 45:
            score -= 0.08  # Getting long

        # Structure bonus
        if has_good_hook:
            score += 0.15  # Strong hook = less dropout at 0-3s
        if has_payoff:
            score += 0.15  # Payoff = less dropout at 10-30s
        if has_cta:
            score += 0.1  # CTA = boost to end retention

        return min(1.0, max(0.0, score))

    @classmethod
    def score_structure_adherence(cls, segments: List[Dict]) -> Dict[str, float]:
        """
        Check if segments follow optimal retention structure.

        Returns dict with structure scores per segment.
        """
        scores = {}

        for i, seg in enumerate(segments):
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            duration = end - start

            structure_score = cls.predict_retention(duration, has_good_hook=True, has_payoff=True)
            scores[i] = structure_score

        return scores


class OptimalLengthCalculator:
    """Calculates optimal clip length for monetization."""

    # Target 25-40s for maximum retention + ad inventory
    TARGET_MIN = 25
    TARGET_MAX = 40
    IDEAL_MID = 32  # Sweet spot

    @classmethod
    def calculate_optimal_length(cls, source_segment_duration: float) -> Tuple[float, str]:
        """
        Determine optimal clip length from source segment.

        Args:
            source_segment_duration: Duration of source segment in seconds

        Returns:
            (optimal_length, reason)
        """
        if source_segment_duration < cls.TARGET_MIN:
            # Pad with context before/after
            return min(cls.TARGET_MIN, source_segment_duration), "Too short - extend if possible"

        elif source_segment_duration <= cls.IDEAL_MID:
            # Perfect length
            return source_segment_duration, "Optimal length"

        elif source_segment_duration <= cls.TARGET_MAX:
            # Good length, maybe trim slightly
            return source_segment_duration, "Good length, consider minor trim"

        else:
            # Too long - suggest split into 2 Shorts
            return None, f"Too long ({source_segment_duration}s) - consider splitting into 2 Shorts"

    @classmethod
    def suggest_split(cls, source_segment_duration: float, midpoint_text: Optional[str] = None) -> Optional[Tuple[float, float]]:
        """
        If segment is too long, suggest how to split it into 2 Shorts.

        Returns (clip1_duration, clip2_duration) or None
        """
        if source_segment_duration <= cls.TARGET_MAX:
            return None

        # Rough split
        per_clip = source_segment_duration / 2
        if per_clip < cls.TARGET_MIN:
            return None  # Can't split evenly

        return (per_clip, per_clip)


class ValueDensityScorer:
    """
    Scores "value per second" - how much content is packed in vs. dead air
    and filler ("um, so, like, you know"). Maps to optimizeclip.md's
    "cut dead space: every second should earn the next second".
    """

    FILLER_WORDS = [
        'um', 'uh', 'like', 'you know', 'i mean', 'basically', 'so like',
        'kind of', 'sort of', 'actually just', 'literally just',
    ]

    IDEAL_WORDS_PER_SEC = (2.0, 3.3)  # Natural, dense speaking pace

    @classmethod
    def score_density(cls, text: str, duration: float) -> float:
        """
        Score 0-1: higher = tighter, denser, less filler-laden speech.
        """
        if not text or duration <= 0:
            return 0.5  # Unknown - neutral

        words = text.split()
        word_count = len(words)
        wps = word_count / duration

        score = 0.5

        lo, hi = cls.IDEAL_WORDS_PER_SEC
        if lo <= wps <= hi:
            score += 0.25
        elif wps < lo * 0.5 or wps > hi * 1.6:
            score -= 0.2  # Either padded with silence or word-vomit
        else:
            score += 0.1

        text_lower = f" {text.lower()} "
        filler_hits = sum(1 for phrase in cls.FILLER_WORDS if f" {phrase} " in text_lower)
        if filler_hits:
            score -= min(0.3, filler_hits * 0.1)

        return min(1.0, max(0.0, score))


class LoopPotentialScorer:
    """
    Scores whether a clip's ending flows back into its opening, encouraging
    rewatches - most valuable on short (<=25s) clips where a seamless loop
    can push completion rate above 100%.
    """

    @classmethod
    def score_loop(cls, text: str, duration: float) -> float:
        if not text:
            return 0.5

        words = [w.strip('.,!?"\'').lower() for w in text.split() if w.strip('.,!?"\'')]
        if len(words) < 6:
            return 0.5

        opening = set(words[:4])
        ending = set(words[-4:])
        echoes = opening & ending

        if duration <= 25:
            return 0.9 if echoes else 0.4  # Loops matter most for tight clips
        return 0.7 if echoes else 0.6  # Nice-to-have on longer clips


class SeriesDetector:
    """Detects if content would benefit from being part of a series."""

    SERIES_PATTERNS = {
        'episode_format': {
            'patterns': ['#1', '#2', 'part 1', 'part 2', 'episode', 'ep '],
            'series_name': 'Natural Episode Series'
        },
        'tip_format': {
            'patterns': ['tip:', 'trick:', 'mistake:', 'how to:', 'python'],
            'series_name': 'Tips & Tricks Series'
        },
        'before_after': {
            'patterns': ['before', 'after', 'wrong', 'right', 'bad', 'good'],
            'series_name': 'Before/After Series'
        },
        'list_format': {
            'patterns': ['ways to', 'reasons', 'mistakes', 'benefits', 'top 5'],
            'series_name': 'List/Countdown Series'
        }
    }

    @classmethod
    def detect_series_potential(cls, text: str, title: Optional[str] = None) -> Tuple[str, str]:
        """
        Detect if clip belongs to a series.

        Returns (series_potential, series_type)
        """
        full_text = f"{title or ''} {text}".lower()

        for series_type, config in cls.SERIES_PATTERNS.items():
            for pattern in config['patterns']:
                if pattern in full_text:
                    return "high", config['series_name']

        # Check for generic repeatable topic
        generic_topics = ['mistake', 'tip', 'how to', 'why', 'code', 'python', 'ai', 'ml']
        for topic in generic_topics:
            if topic in full_text:
                return "medium", "Expandable Topic Series"

        return "low", "One-off content"


class MonetizationScorer:
    """Combines all monetization metrics into single score."""

    # Weights for final monetization score
    WEIGHT_HOOK = 0.25  # Hook strength very important
    WEIGHT_VVSA = 0.30  # VVSA is primary signal (first 1-3s)
    WEIGHT_RETENTION = 0.25  # Overall retention
    WEIGHT_LENGTH = 0.05  # Optimal length
    WEIGHT_STRUCTURE = 0.05  # Following good template
    WEIGHT_VALUE_DENSITY = 0.05  # Dead-space / filler penalty
    WEIGHT_LOOP = 0.05  # Loop-friendly ending (biggest win on short clips)

    @classmethod
    def score_clip(
        cls,
        segment: Dict,
        has_face: Optional[bool] = None,
        has_motion: Optional[bool] = None,
    ) -> MonetizationMetrics:
        """
        Comprehensive monetization scoring of a clip.

        Args:
            segment: Dict with text, start, end, etc. May also carry
                'has_face' / 'has_motion' keys as a fallback if the explicit
                kwargs below are not provided.
            has_face: Is a face visible in the first ~3s? None = unknown
                (no detector run yet) -> treated as neutral, NOT as False.
            has_motion: Is there motion/cuts in the first ~3s? None = unknown
                (no detector run yet) -> treated as neutral, NOT as False.

        Returns:
            MonetizationMetrics with all scores
        """
        text = segment.get('text', '')
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        duration = end - start

        # Fall back to segment-embedded detection results if not passed explicitly.
        if has_face is None:
            has_face = segment.get('has_face')
        if has_motion is None:
            has_motion = segment.get('has_motion')

        # Component scores
        hook_score, hook_reason = HookStrengthDetector.detect_hook_strength(text)
        vvsa_score = VVSAPredictor.predict_vvsa(text, has_motion=has_motion, has_face=has_face)
        retention_score = RetentionCurveOptimizer.predict_retention(duration, has_good_hook=hook_score > 0.6)

        optimal_len, len_reason = OptimalLengthCalculator.calculate_optimal_length(duration)
        structure_score = RetentionCurveOptimizer.predict_retention(duration, has_good_hook=True, has_payoff=True)

        series_potential, series_type = SeriesDetector.detect_series_potential(text)

        value_density_score = ValueDensityScorer.score_density(text, duration)
        loop_score = LoopPotentialScorer.score_loop(text, duration)

        # Combined monetization score
        monetization_score = (
            cls.WEIGHT_HOOK * hook_score +
            cls.WEIGHT_VVSA * vvsa_score +
            cls.WEIGHT_RETENTION * retention_score +
            cls.WEIGHT_LENGTH * (optimal_len / 40 if optimal_len else 0.3) +
            cls.WEIGHT_STRUCTURE * structure_score +
            cls.WEIGHT_VALUE_DENSITY * value_density_score +
            cls.WEIGHT_LOOP * loop_score
        )

        # Recommendations
        recommendations = []
        if hook_score < 0.6:
            recommendations.append(f"⚠ Weak hook: {hook_reason}")
        if vvsa_score < 0.7:
            recommendations.append("⚠ Might lose viewers in first 3s - strengthen opening")
        if retention_score < 0.73:
            recommendations.append("⚠ Structure could hurt retention - add payoff sooner")
        if duration > OptimalLengthCalculator.TARGET_MAX:
            recommendations.append(f"💡 Consider splitting into 2 Shorts ({duration:.0f}s too long)")
        if series_potential in ["high", "medium"]:
            recommendations.append(f"💡 Great for series: {series_type}")
        if value_density_score < 0.5:
            recommendations.append("⚠ Dead space or filler words detected - tighten the cut")
        if duration <= 25 and loop_score < 0.6:
            recommendations.append("💡 Short clip - end on a word/phrase that echoes the hook for a loop")
        if has_face is None or has_motion is None:
            recommendations.append(
                "ℹ️ Face/motion data unavailable — VVSA estimate based on text signals only"
            )

        return MonetizationMetrics(
            hook_strength=hook_score,
            vvsa_score=vvsa_score,
            retention_score=retention_score,
            optimal_length=optimal_len or duration,
            structure_score=structure_score,
            monetization_score=min(1.0, monetization_score),
            series_potential=series_potential,
            recommendations=recommendations,
            has_face=has_face,
            has_motion=has_motion,
            value_density_score=value_density_score,
            loop_score=loop_score,
        )

    @classmethod
    def batch_score(cls, segments: List[Dict]) -> List[Dict]:
        """Score multiple segments."""
        scored = []
        for segment in segments:
            metrics = cls.score_clip(segment)
            segment_scored = segment.copy()
            segment_scored['monetization'] = {
                'hook_strength': metrics.hook_strength,
                'vvsa_score': metrics.vvsa_score,
                'retention_score': metrics.retention_score,
                'optimal_length': metrics.optimal_length,
                'structure_score': metrics.structure_score,
                'monetization_score': metrics.monetization_score,
                'series_potential': metrics.series_potential,
                'recommendations': metrics.recommendations,
                'has_face': metrics.has_face,
                'has_motion': metrics.has_motion,
                'value_density_score': metrics.value_density_score,
                'loop_score': metrics.loop_score,
            }
            scored.append(segment_scored)
        return scored

    @classmethod
    def print_clip_analysis(cls, metrics: MonetizationMetrics, title: str = ""):
        """Pretty-print clip analysis."""
        print("\n" + "=" * 70)
        if title:
            print(f"📊 {title}")
        print("=" * 70)
        print(f"Hook Strength:       {metrics.hook_strength:.1%}  {'🔥' if metrics.hook_strength > 0.7 else '⚠️' if metrics.hook_strength > 0.5 else '❌'}")
        print(f"VVSA (No Swipe):     {metrics.vvsa_score:.1%}  {'✓ Excellent' if metrics.vvsa_score > 0.8 else '✓ Good' if metrics.vvsa_score > 0.7 else '⚠️ Risky'}")
        print(f"Predicted Retention: {metrics.retention_score:.1%}  {'🎯' if metrics.retention_score > 0.73 else '📉'}")
        print(f"Optimal Length:      {metrics.optimal_length:.0f}s")
        print(f"Structure Fit:       {metrics.structure_score:.1%}")
        print(f"\n💰 MONETIZATION:     {metrics.monetization_score:.1%}  {'🤑 High' if metrics.monetization_score > 0.7 else '💵 Medium' if metrics.monetization_score > 0.5 else '❌ Low'}")
        print(f"Series Potential:    {metrics.series_potential.upper()}")
        if metrics.recommendations:
            print(f"\n📝 Recommendations:")
            for rec in metrics.recommendations:
                print(f"   {rec}")
        print("=" * 70 + "\n")
