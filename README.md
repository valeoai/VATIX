# How Far Can 5,500 Hours of Driving Take You? A Scaling Law Analysis of Video Diffusion Models

[![Workshop](https://img.shields.io/badge/Workshop-DriveX%20%40%20ECCV%202026-blue)](#)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](#)
[![Models](https://img.shields.io/badge/Models-Coming%20Soon-orange)](#)

> **Accepted at the 6th DriveX Workshop in conjunction with ECCV 2026**

This is the official repository for the paper **"How Far Can 5,500 Hours of Driving Take You? A Scaling Law Analysis of Video Diffusion Models"**. 

**Authors:** Victor Besnier, Anh-Quan Cao, Elias Ramzi, Spyros Gidaris, Tuan-Hung Vu, Andrei Bursuc, Eloi Zablocki, and Matthieu Cord.
**Affiliation:** Valeo, Valeo.ai, Paris.

---

## 🚧 Status: Code and Models Coming Soon!
**The code and pretrained models for the VATIX model family will be publicly released shortly**. Please star or watch this repository for updates.

## 📖 Overview
Video generation for autonomous driving cannot follow the web-scale route: driving data is expensive to collect, bound by privacy requirements, and cannot be scraped at will, meaning models must make the most of a fixed corpus. 

We present a systematic scaling-law study of video diffusion models trained from scratch on driving data. We trained a family of Diffusion Transformer (DiT) flow-matching models ranging from 1.6M to 9B parameters, using up to 5,500 hours of driving data. 

### Key Findings
* **Consistent Scaling Laws:** Validation loss follows consistent power laws in both model size and training exposure. 
* **Compute Optimization:** Loss improves much faster with training exposure than with model size, making longer training the most effective way to improve a fixed model under limited compute.
* **The Value of Scale:** Larger models continue to achieve lower asymptotic loss, meaning compute-optimal scaling still favors increasing model size when sufficient compute and data are available.
---
