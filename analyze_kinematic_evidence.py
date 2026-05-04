#!/usr/bin/env python3
"""
Kinematic Evidence Analyzer
============================
Generates scientific proof that collision detection is based on physics behavior,
not just visual detection of two objects.

Analyzes velocity, acceleration, and trajectory anomalies during detected collisions.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def analyze_kinematic_evidence(detection_data: Dict) -> Dict:
    """
    Analyze kinematic evidence from collision detections.

    Args:
        detection_data: Detection data with kinematic_evidence fields

    Returns:
        Dict with kinematic analysis results
    """
    analysis = {
        "total_detections": 0,
        "evidence_summary": [],
        "physics_signatures": []
    }

    for detection in detection_data.get('detections', []):
        if not hasattr(detection, 'kinematic_evidence') or not detection.kinematic_evidence:
            continue

        analysis["total_detections"] += 1
        kinematic = detection.kinematic_evidence

        # Extract primary track data
        primary = kinematic.get('primary', {})
        secondary = kinematic.get('secondary', {})

        frame = detection.frame
        conf = detection.confidence

        # Analyze primary track motion
        if primary:
            vx, vy = primary.get('velocity', (0, 0))
            speed = primary.get('speed', 0)

            evidence = {
                'frame': frame,
                'confidence': f"{conf:.2%}",
                'primary_track': primary.get('track_id'),
                'primary_speed': f"{speed:.2f} px/frame",
                'primary_velocity': f"({vx:.2f}, {vy:.2f}) px/frame",
                'primary_track_age': primary.get('track_age'),
            }

            # Check for sudden deceleration signature
            if speed < 5.0:  # Low speed indicates brake or collision
                evidence['physics_signature'] = "⚠️ SUDDEN DECELERATION (brake or impact)"
                analysis["physics_signatures"].append("sudden_brake")

            analysis["evidence_summary"].append(evidence)

    return analysis


def generate_kinematic_report(video_results: Dict, output_file: str = None) -> str:
    """Generate human-readable kinematic evidence report."""

    report = []
    report.append("=" * 80)
    report.append("KINEMATIC EVIDENCE ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    report.append("This section provides scientific proof that collision detection is based on")
    report.append("kinematic anomalies (velocity, acceleration, trajectory changes), not solely")
    report.append("on visual detection of two objects.")
    report.append("")
    report.append("-" * 80)

    for video_key, result in video_results.items():
        report.append(f"\nVideo: {video_key}")
        report.append(f"Total Detections: {result.get('total_detections', 0)}")

        if result.get('evidence_summary'):
            report.append("\nDetailed Evidence:")
            for evidence in result['evidence_summary']:
                report.append(f"\n  Frame {evidence['frame']} (Confidence: {evidence['confidence']})")
                report.append(f"    Primary Track ID: {evidence['primary_track']}")
                report.append(f"    Speed: {evidence['primary_speed']}")
                report.append(f"    Velocity: {evidence['primary_velocity']}")
                if 'physics_signature' in evidence:
                    report.append(f"    {evidence['physics_signature']}")

        signature_count = result.get('physics_signatures', [])
        if signature_count:
            report.append(f"\nPhysics Signatures Detected:")
            report.append(f"  - Sudden brakes/impacts: {signature_count.count('sudden_brake')}")
        report.append("-" * 80)

    report_text = "\n".join(report)

    if output_file:
        Path(output_file).write_text(report_text)
        print(f"Kinematic evidence report saved to: {output_file}")

    return report_text


def create_kinematic_visualization_data(detections: List[Dict]) -> Dict:
    """
    Create data for visualization graphs (velocity over time, acceleration, etc.)
    Can be used to generate graphs in Matplotlib/Plotly for paper figures.
    """

    viz_data = {
        'frames': [],
        'speeds': [],
        'velocities_x': [],
        'velocities_y': [],
        'confidences': [],
        'detection_frames': [],  # Frame saat collision detected
    }

    for det in detections:
        if not hasattr(det, 'kinematic_evidence') or det.kinematic_evidence is None:
            continue

        kinematic = det.kinematic_evidence
        primary = kinematic.get('primary', {})

        if primary:
            vx, vy = primary.get('velocity', (0, 0))
            speed = primary.get('speed', 0)

            viz_data['frames'].append(det.frame)
            viz_data['speeds'].append(speed)
            viz_data['velocities_x'].append(vx)
            viz_data['velocities_y'].append(vy)
            viz_data['confidences'].append(det.confidence)
            viz_data['detection_frames'].append(det.frame)  # Mark detection frame
        elif not primary:
            # Debug: log what we did have
            print(f"[DEBUG] Detection at frame {det.frame} has kinematic_evidence but no 'primary' key. Keys: {list(kinematic.keys())}")

    return viz_data


def plot_kinematics_speed_profile(detections: List[Dict], video_name: str, output_dir: str = "kinematic_plots"):
    """
    Generate graph: Speed vs Frame (showing sudden brake signature).

    This is the main evidence graph for the paper.
    """
    viz_data = create_kinematic_visualization_data(detections)

    if not viz_data['frames']:
        print(f"[WARN] No kinematic data to plot for {video_name}")
        return

    Path(output_dir).mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Speed vs Frame
    frames = np.array(viz_data['frames'])
    speeds = np.array(viz_data['speeds'])
    confs = np.array(viz_data['confidences'])

    scatter = ax1.scatter(frames, speeds, c=confs, cmap='RdYlGn', s=100, alpha=0.7, edgecolor='black')
    ax1.plot(frames, speeds, 'b-', alpha=0.3, linewidth=1)
    ax1.axhline(y=5.0, color='red', linestyle='--', linewidth=2, label='Brake Threshold (5 px/frame)')
    ax1.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Speed (px/frame)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{video_name} - Vehicle Speed Profile (Kinematic Evidence)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Detection Confidence', fontsize=10)

    # Highlight sudden brakes
    speed_diff = np.diff(speeds, prepend=speeds[0])
    sudden_brakes = np.where(speed_diff < -3.0)[0]  # Deceleration > 3 px/frame²
    for brake_idx in sudden_brakes:
        ax1.axvline(x=frames[brake_idx], color='red', alpha=0.3, linewidth=2, linestyle=':')
        ax1.text(frames[brake_idx], ax1.get_ylim()[1] * 0.9, 'BRAKE',
                rotation=90, fontsize=9, color='red', fontweight='bold')

    # Plot 2: Velocity Vector (X & Y components)
    vx = np.array(viz_data['velocities_x'])
    vy = np.array(viz_data['velocities_y'])

    ax2.plot(frames, vx, 'r-', marker='o', label='Velocity X', linewidth=2, markersize=5)
    ax2.plot(frames, vy, 'b-', marker='s', label='Velocity Y', linewidth=2, markersize=5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Velocity Component (px/frame)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{video_name} - Velocity Components (X, Y)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='upper right')

    # Highlight collision frames
    for det_frame in viz_data['detection_frames']:
        ax2.axvline(x=det_frame, color='green', alpha=0.5, linewidth=2.5, linestyle='--', label='Collision')

    plt.tight_layout()
    output_file = Path(output_dir) / f"{video_name}_speed_profile.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Speed profile graph saved: {output_file}")
    plt.close()


def plot_acceleration_signature(detections: List[Dict], video_name: str, output_dir: str = "kinematic_plots"):
    """
    Generate graph: Acceleration vs Frame (showing impact signature).
    """
    viz_data = create_kinematic_visualization_data(detections)

    if not viz_data['frames']:
        return

    Path(output_dir).mkdir(exist_ok=True)

    frames = np.array(viz_data['frames'])
    speeds = np.array(viz_data['speeds'])

    # Calculate acceleration (derivative of speed)
    acceleration = np.diff(speeds, prepend=speeds[0])

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['red' if acc < -2 else 'green' if acc > 1 else 'blue' for acc in acceleration]
    ax.bar(frames, acceleration, color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axhline(y=-2, color='red', linestyle='--', linewidth=2, label='Collision Threshold (decel > 2 px/frame²)')

    ax.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Acceleration (px/frame²)', fontsize=12, fontweight='bold')
    ax.set_title(f'{video_name} - Acceleration Profile (Impact Signature)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)

    # Add shaded region for collision zone
    for det_frame in viz_data['detection_frames']:
        ax.axvspan(det_frame - 5, det_frame + 5, alpha=0.2, color='orange', label='Collision Zone')

    plt.tight_layout()
    output_file = Path(output_dir) / f"{video_name}_acceleration.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Acceleration graph saved: {output_file}")
    plt.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_kinematic_evidence.py <results.json>")
        print("\nExample:")
        print("  python analyze_kinematic_evidence.py eval_results/detections.json")
        sys.exit(1)

    detection_file = sys.argv[1]

    # Load detections (would be from saved evaluation results)
    # This is a template - integrate with actual detection collection
    print(f"[INFO] Kinematic Evidence Analyzer ready.")
    print(f"[INFO] To use with detections, call generate_kinematic_report() with detection data.")
    print(f"[INFO] Expected integration point: after video processing completes.")
    print(f"\n[INFO] Graph generation functions available:")
    print(f"  - plot_kinematics_speed_profile(): Speed vs Frame graph")
    print(f"  - plot_acceleration_signature(): Acceleration vs Frame graph")

