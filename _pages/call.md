---
layout: page
title: call for blogposts
permalink: /call
description:
nav: true
nav_order: 2
---

# Call for blogposts

We invite all researchers and practicioners to submit a blogpost discussing work previously published at ICLR, to the ICLR 2023 blogpost track.

The format and process for this blog post track is as follows:

- Write a post on a subject that has been published at ICLR relatively recently, with the constraint that one cannot write a blog post on work that they have a conflict of interest with.
    This implies that one cannot review their own work, or work originating from their institution or company.
    We want to foster productive discussion about ideas, and prevent posts that intentionally aim to help or hurt individuals or institutions.

- The posts will be created and published under a unified template; see [the submission instructions]({{ '/submitting/' | relative_url }})
    and the [sample post]({{ '/blog/2022/distill-example' | relative_url }}) hosted on the blog of this website.

- Blogs will be peer-reviewed (double-blind) for quality and novelty of the content: clarity and pedagogy of the exposition, new theoretical or practical insights, reproduction/extension of experiments, etc.
We are slightly relaxing the double-blind constraints by assuming good faith from both submitters and reviewers (see [the submission instructions]({{ '/submitting/' | relative_url }}) for more details).

## Key Dates

- **Submission deadline**: February 2nd, 2023
&nbsp;

- **Notification of acceptance**: March 31st, 2023
&nbsp;

- **Camera-ready merge**: *TBD*

## Submission Guidelines

> See [the submission instructions]({{ '/submitting/' | relative_url }}) for more details.

For this edition of the Blogposts Track, we will forgo the requirement for total anonymity. 
The blog posts **must be anonymized for the review process**, but users will submit their anonymized blog posts via a pull request to a staging repository (in addition to a submission on OpenReview).
The post will be merged into the staging repository, where it will be deployed to a separate Github Pages website. 
Reviewers will be able to access the posts directly through a public url on this staging website, and will submit their reviews on OpenReview.
Reviewers should refrain from looking at the git history for the post, which may reveal information about the authors.

This still largely follows the Double-Blind reviewing principle; it is no less double-blind than when reviewers are asked to score papers that have previously been released to [arXiv](https://arxiv.org/), an overwhelmingly common practice in the ML community.
This approach was chosen to lower the burden on both the organizers and the authors; last year, many submissions had to be reworked once deployed due to a variety of reasons.
By allowing the authors to render their websites to Github Pages prior to the review process, we hope to avoid this issue entirely. 
We also avoid the issue of having to host the submissions on a separate server during the reviewing process.

However, we understand the desire for total anonymity. 
Authors that wish to have a fully double-blind process might consider creating new GitHub accounts without identifying information which will only be used for this track.