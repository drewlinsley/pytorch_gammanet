"""Analysis tools for in silico neurophysiology experiments.

This module provides classes to analyze neural responses extracted from
GammaNet models, including orientation tuning, contrast response functions,
and surround modulation effects.
"""

import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class TuningCurve:
    """Container for tuning curve data and fits."""
    x_values: np.ndarray  # Stimulus values (e.g., orientations)
    y_values: np.ndarray  # Response values
    y_err: Optional[np.ndarray] = None  # Standard errors
    fit_params: Optional[Dict] = None
    fit_function: Optional[str] = None
    preferred_value: Optional[float] = None
    bandwidth: Optional[float] = None
    modulation_index: Optional[float] = None
    r_squared: Optional[float] = None


class OrientationTuningAnalyzer:
    """Analyze orientation tuning properties."""
    
    @staticmethod
    def von_mises(x, amplitude, kappa, mu, baseline):
        """Von Mises function for orientation tuning.
        
        Args:
            x: Orientations in radians
            amplitude: Response amplitude
            kappa: Concentration parameter (inverse width)
            mu: Preferred orientation
            baseline: Baseline response
        """
        return baseline + amplitude * np.exp(kappa * (np.cos(2 * (x - mu)) - 1))
    
    @staticmethod
    def gaussian(x, amplitude, sigma, mu, baseline):
        """Gaussian function for orientation tuning."""
        return baseline + amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    def fit_tuning_curve(self,
                        orientations: np.ndarray,
                        responses: np.ndarray,
                        fit_type: str = "von_mises") -> TuningCurve:
        """Fit orientation tuning curve.
        
        Args:
            orientations: Stimulus orientations in degrees
            responses: Neural responses
            fit_type: Type of fit - "von_mises" or "gaussian"
            
        Returns:
            TuningCurve object with fit results
        """
        # Convert to radians for von Mises
        orientations_rad = np.radians(orientations)
        
        # Initial parameter guesses
        baseline = np.min(responses)
        amplitude = np.max(responses) - baseline
        pref_idx = np.argmax(responses)
        pref_ori = orientations_rad[pref_idx]
        
        try:
            if fit_type == "von_mises":
                # Fit von Mises function
                p0 = [amplitude, 1.0, pref_ori, baseline]
                bounds = ([0, 0, 0, 0], [np.inf, 10, 2*np.pi, np.inf])
                
                popt, pcov = curve_fit(
                    self.von_mises, orientations_rad, responses,
                    p0=p0, bounds=bounds, maxfev=5000
                )
                
                # Calculate goodness of fit
                y_fit = self.von_mises(orientations_rad, *popt)
                
                # Extract parameters
                fit_params = {
                    "amplitude": popt[0],
                    "kappa": popt[1],
                    "preferred_orientation": np.degrees(popt[2]) % 180,
                    "baseline": popt[3]
                }
                
                # Calculate bandwidth (HWHM)
                bandwidth = np.degrees(np.arccos(1 + np.log(0.5) / popt[1]) / 2)
                
            else:  # gaussian
                # Use wrapped Gaussian for circular data
                p0 = [amplitude, 30, orientations[pref_idx], baseline]
                
                popt, pcov = curve_fit(
                    self.gaussian, orientations, responses,
                    p0=p0, maxfev=5000
                )
                
                y_fit = self.gaussian(orientations, *popt)
                
                fit_params = {
                    "amplitude": popt[0],
                    "sigma": popt[1],
                    "preferred_orientation": popt[2] % 180,
                    "baseline": popt[3]
                }
                
                bandwidth = popt[1]  # sigma as bandwidth measure
                
            # Calculate R-squared
            ss_res = np.sum((responses - y_fit) ** 2)
            ss_tot = np.sum((responses - np.mean(responses)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate modulation index
            modulation_index = (np.max(responses) - np.min(responses)) / (np.max(responses) + np.min(responses))
            
        except Exception as e:
            warnings.warn(f"Fitting failed: {e}")
            fit_params = None
            bandwidth = None
            r_squared = None
            modulation_index = None
            
        return TuningCurve(
            x_values=orientations,
            y_values=responses,
            fit_params=fit_params,
            fit_function=fit_type,
            preferred_value=fit_params["preferred_orientation"] if fit_params else orientations[pref_idx],
            bandwidth=bandwidth,
            modulation_index=modulation_index,
            r_squared=r_squared
        )
    
    def compute_orientation_selectivity_index(self,
                                            responses: np.ndarray,
                                            orientations: np.ndarray) -> float:
        """Compute orientation selectivity index (OSI).
        
        OSI = (R_pref - R_ortho) / (R_pref + R_ortho)
        """
        pref_idx = np.argmax(responses)
        pref_ori = orientations[pref_idx]
        
        # Find orthogonal orientation
        ortho_ori = (pref_ori + 90) % 180
        ortho_idx = np.argmin(np.abs(orientations - ortho_ori))
        
        r_pref = responses[pref_idx]
        r_ortho = responses[ortho_idx]
        
        if r_pref + r_ortho > 0:
            osi = (r_pref - r_ortho) / (r_pref + r_ortho)
        else:
            osi = 0
            
        return osi
    
    def compute_direction_selectivity_index(self,
                                          responses: np.ndarray,
                                          directions: np.ndarray) -> float:
        """Compute direction selectivity index (DSI).
        
        DSI = (R_pref - R_null) / (R_pref + R_null)
        where null is 180 degrees opposite to preferred
        """
        pref_idx = np.argmax(responses)
        pref_dir = directions[pref_idx]
        
        # Find null direction (opposite)
        null_dir = (pref_dir + 180) % 360
        null_idx = np.argmin(np.abs(directions - null_dir))
        
        r_pref = responses[pref_idx]
        r_null = responses[null_idx]
        
        if r_pref + r_null > 0:
            dsi = (r_pref - r_null) / (r_pref + r_null)
        else:
            dsi = 0
            
        return dsi


class ContrastResponseAnalyzer:
    """Analyze contrast response functions."""
    
    @staticmethod
    def naka_rushton(contrast, r_max, c50, n, baseline=0):
        """Naka-Rushton contrast response function.
        
        Args:
            contrast: Stimulus contrast (0-1)
            r_max: Maximum response
            c50: Semi-saturation contrast
            n: Exponent (typically ~2)
            baseline: Baseline response
        """
        return baseline + r_max * (contrast ** n) / (contrast ** n + c50 ** n)
    
    def fit_contrast_response(self,
                            contrasts: np.ndarray,
                            responses: np.ndarray) -> TuningCurve:
        """Fit contrast response function.
        
        Args:
            contrasts: Stimulus contrasts (0-1)
            responses: Neural responses
            
        Returns:
            TuningCurve object with fit results
        """
        # Initial parameter guesses
        r_max = np.max(responses)
        c50 = 0.3  # Typical value
        n = 2.0
        baseline = np.min(responses)
        
        try:
            p0 = [r_max, c50, n, baseline]
            bounds = ([0, 0, 0.5, 0], [np.inf, 1, 5, np.inf])
            
            popt, pcov = curve_fit(
                self.naka_rushton, contrasts, responses,
                p0=p0, bounds=bounds, maxfev=5000
            )
            
            # Calculate goodness of fit
            y_fit = self.naka_rushton(contrasts, *popt)
            ss_res = np.sum((responses - y_fit) ** 2)
            ss_tot = np.sum((responses - np.mean(responses)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            fit_params = {
                "r_max": popt[0],
                "c50": popt[1],
                "n": popt[2],
                "baseline": popt[3]
            }
            
        except Exception as e:
            warnings.warn(f"Fitting failed: {e}")
            fit_params = None
            r_squared = None
            
        return TuningCurve(
            x_values=contrasts,
            y_values=responses,
            fit_params=fit_params,
            fit_function="naka_rushton",
            r_squared=r_squared
        )
    
    def compute_contrast_gain(self, crf: TuningCurve) -> Optional[float]:
        """Compute contrast gain (slope at c50)."""
        if crf.fit_params is None:
            return None
            
        c50 = crf.fit_params["c50"]
        r_max = crf.fit_params["r_max"]
        n = crf.fit_params["n"]
        
        # Derivative at c50
        gain = (n * r_max) / (2 * c50)
        return gain


class SurroundModulationAnalyzer:
    """Analyze surround modulation effects."""
    
    def compute_suppression_index(self,
                                 center_only: float,
                                 center_surround: float) -> float:
        """Compute surround suppression index.
        
        SI = (center_only - center_surround) / center_only
        """
        if center_only > 0:
            return (center_only - center_surround) / center_only
        return 0
    
    def analyze_orientation_tuning_shift(self,
                                       center_responses: np.ndarray,
                                       surround_responses: np.ndarray,
                                       orientations: np.ndarray) -> Dict:
        """Analyze how surround changes orientation tuning.
        
        Returns:
            Dictionary with shift metrics
        """
        # Fit tuning curves
        analyzer = OrientationTuningAnalyzer()
        center_tuning = analyzer.fit_tuning_curve(orientations, center_responses)
        surround_tuning = analyzer.fit_tuning_curve(orientations, surround_responses)
        
        results = {
            "preferred_shift": 0,
            "bandwidth_change": 0,
            "amplitude_change": 0,
            "suppression_index": 0
        }
        
        if center_tuning.fit_params and surround_tuning.fit_params:
            # Preferred orientation shift
            results["preferred_shift"] = (
                surround_tuning.preferred_value - center_tuning.preferred_value
            )
            
            # Bandwidth change
            if center_tuning.bandwidth and surround_tuning.bandwidth:
                results["bandwidth_change"] = (
                    surround_tuning.bandwidth - center_tuning.bandwidth
                )
            
            # Amplitude change
            results["amplitude_change"] = (
                surround_tuning.fit_params["amplitude"] - 
                center_tuning.fit_params["amplitude"]
            )
            
            # Overall suppression
            results["suppression_index"] = self.compute_suppression_index(
                center_tuning.fit_params["amplitude"],
                surround_tuning.fit_params["amplitude"]
            )
            
        return results
    
    def analyze_collinear_facilitation(self,
                                     target_only: np.ndarray,
                                     target_flanker: np.ndarray,
                                     flanker_distances: np.ndarray) -> Dict:
        """Analyze collinear facilitation effects.
        
        Args:
            target_only: Responses to target alone
            target_flanker: Responses to target + flankers
            flanker_distances: Distances of flankers
            
        Returns:
            Analysis results
        """
        # Compute facilitation index at each distance
        facilitation = np.zeros_like(flanker_distances, dtype=float)
        
        for i, dist in enumerate(flanker_distances):
            if target_only[i] > 0:
                facilitation[i] = (target_flanker[i] - target_only[i]) / target_only[i]
            
        # Find optimal distance
        optimal_idx = np.argmax(facilitation)
        optimal_distance = flanker_distances[optimal_idx]
        max_facilitation = facilitation[optimal_idx]
        
        # Fit spatial tuning of facilitation
        try:
            # Fit Gaussian to facilitation profile
            from scipy.optimize import curve_fit
            
            def gaussian(x, amp, mu, sigma):
                return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            
            popt, _ = curve_fit(
                gaussian, flanker_distances, facilitation,
                p0=[max_facilitation, optimal_distance, 10]
            )
            
            spatial_extent = popt[2]  # Sigma of Gaussian
            
        except:
            spatial_extent = None
            
        return {
            "facilitation_profile": facilitation,
            "optimal_distance": optimal_distance,
            "max_facilitation": max_facilitation,
            "spatial_extent": spatial_extent
        }


class PopulationCodingAnalyzer:
    """Analyze population coding properties."""
    
    def decode_orientation(self,
                          population_responses: np.ndarray,
                          neuron_preferences: np.ndarray,
                          method: str = "vector_average") -> float:
        """Decode orientation from population response.
        
        Args:
            population_responses: Responses of all neurons
            neuron_preferences: Preferred orientation of each neuron
            method: Decoding method - "vector_average" or "maximum_likelihood"
            
        Returns:
            Decoded orientation
        """
        if method == "vector_average":
            # Convert to radians for circular averaging
            prefs_rad = np.radians(neuron_preferences * 2)  # Double for 180° periodicity
            
            # Compute vector average
            x = np.sum(population_responses * np.cos(prefs_rad))
            y = np.sum(population_responses * np.sin(prefs_rad))
            
            # Convert back to degrees
            decoded = np.degrees(np.arctan2(y, x) / 2) % 180
            
        elif method == "maximum_likelihood":
            # Simplified ML decoding
            # Weight each neuron's preference by its response
            weights = population_responses / np.sum(population_responses)
            decoded = np.sum(weights * neuron_preferences) % 180
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return decoded
    
    def compute_fisher_information(self,
                                 tuning_curves: List[TuningCurve],
                                 stimulus_value: float) -> float:
        """Compute Fisher information at a stimulus value.
        
        Measures how well the population can discriminate small changes
        around the stimulus value.
        """
        fisher_info = 0
        
        for curve in tuning_curves:
            if curve.fit_params is None:
                continue
                
            # Compute derivative of tuning curve at stimulus value
            # Numerical derivative
            delta = 1.0  # 1 degree
            
            if curve.fit_function == "von_mises":
                f_plus = OrientationTuningAnalyzer.von_mises(
                    np.radians(stimulus_value + delta), **curve.fit_params
                )
                f_minus = OrientationTuningAnalyzer.von_mises(
                    np.radians(stimulus_value - delta), **curve.fit_params
                )
            else:
                continue
                
            derivative = (f_plus - f_minus) / (2 * delta)
            
            # Add to Fisher information (assuming Poisson noise)
            if f_plus > 0:
                fisher_info += derivative ** 2 / f_plus
                
        return fisher_info
    
    def compute_sparseness(self, responses: np.ndarray) -> float:
        """Compute population sparseness (Rolls & Tovee, 1995).
        
        S = 1 - (sum(r_i/n))^2 / sum(r_i^2/n)
        
        Returns value between 0 (dense) and 1 (sparse)
        """
        n = len(responses)
        if n == 0 or np.sum(responses) == 0:
            return 0
            
        mean_response = np.mean(responses)
        mean_squared = np.mean(responses ** 2)
        
        if mean_squared > 0:
            sparseness = 1 - (mean_response ** 2) / mean_squared
        else:
            sparseness = 0
            
        return sparseness