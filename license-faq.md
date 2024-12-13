# License FAQ

This License FAQ elucidates the terms and conditions governing the use of data, models, code, configurations, and documentation provided by the DanLing Team (also known as DanLing) ("we," "us," or "our"). It serves as an addendum to our _[License](license.md)_.

## 0. Summary of Key Points

This summary highlights the key aspects of our license. For more detailed information, please refer to the corresponding sections below or read the full _[License](license.md)_.

<div class="grid cards" markdown>

!!! question "What constitutes the 'source code' in MultiMolecule?"

    The source code in MultiMolecule includes all materials necessary for training the models, such as data, code, configuration files, documentation, and research papers. Model weights are considered object code.

    [:octicons-arrow-right-24: What constitutes the 'source code' in MultiMolecule?](#1-what-constitutes-the-source-code-in-multimolecule)

!!! question "Am I required to open-source my fine-tuned model weights?"

    Yes, any fine-tuned model weights derived from MultiMolecule's pre-trained models must be open-sourced under the _[License](license.md)_.

    [:octicons-arrow-right-24: Am I required to open-source my fine-tuned model weights?](#2-am-i-required-to-open-source-my-fine-tuned-model-weights)

!!! question "Can I publish research papers using MultiMolecule?"

    It depends. You may publish research papers in certain open access and non-profit venues. For closed access or author fee venues, you must obtain a separate license.

    [:octicons-arrow-right-24: Can I publish research papers using MultiMolecule?](#3-can-i-publish-research-papers-using-multimolecule)

!!! question "Can I publish derivative works of MultiMolecule in Nature Machine Intelligence?"

    No. MultiMolecule must not be used in any work submitted to or published by Nature Machine Intelligence.

    [:octicons-arrow-right-24: Publication in Nature Machine Intelligence](#4-publication-in-nature-machine-intelligence)

!!! question "Can I use MultiMolecule for commercial purposes?"

    Yes, you can use MultiMolecule for commercial purposes under the terms of the _[License](license.md)_.

    [:octicons-arrow-right-24: Can I use MultiMolecule for commercial purposes?](#5-can-i-use-multimolecule-for-commercial-purposes)

!!! question "Do people affiliated with certain organizations have specific license terms?"

    Yes, people affiliated with certain organizations have specific license terms.

    [:octicons-arrow-right-24: Do people affiliated with certain organizations have specific license terms?](#6-do-people-affiliated-with-certain-organizations-have-specific-license-terms)

</div>

## 1. What constitutes the "source code" in MultiMolecule?

In the context of MultiMolecule, "source code" refers to all materials necessary for training and developing the models. This includes:

- **Data**: Datasets used for training and evaluating the models.
- **Code**: All scripts, programs, and utilities required for model training, evaluation, and deployment.
- **Configuration**: Configuration files and settings that dictate the behavior of the models and training processes.
- **Documentation**: Manuals, guides, and other forms of documentation that facilitate understanding and using the software.
- **Research Papers**: Manuscripts and publications that describe the methodologies and findings related to MultiMolecule.

Conversely, the **model weights** generated from the training process are considered "object code."

This distinction ensures that all necessary components for training and modifying the models remain accessible, while the trained model weights are treated as object code, aligning with our commitment to providing maximum freedom for the community.

## 2. Am I required to open-source my fine-tuned model?

Yes, if you utilize a pre-trained model from MultiMolecule's model hub and subsequently fine-tune it, you must open-source your modified model weights under the _[License](license.md)_.

Model weights are considered object code in MultiMolecule. Therefore, in addition to sharing the fine-tuned model weights, you must also open-source all materials necessary to reproduce the model, including training data, configuration files, and any scripts used for fine-tuning.

In addition, if you use our code, our data, or our models for any purpose, you are obligated to open-source all related material, including fine-tuned models, modified code, or derived datasets, under the _[License](license.md)_. This approach ensures that all enhancements, modifications, and related resources remain freely available to the community.

## 3. Can I publish research papers using MultiMolecule?

!!! success "Open Access"

    You may publish research papers in diamond open access venues.

You are permitted to publish research papers in fully open access journals, conferences, or platforms that do not charge fees to either authors or readers, provided that all published manuscripts are made available under one of the following licenses that permits the sharing of manuscripts:

- [GNU Free Documentation License (GFDL)](https://www.gnu.org/licenses/fdl.html)
- [Creative Commons Licenses](https://creativecommons.org/licenses/)
- [OSI-Approved Licenses](https://opensource.org/licenses)

!!! warning "Non-Profit"

    You may publish research papers in certain non-profit venues.

You are permitted to publish research papers in certain non-profit journals, conferences, or platforms. Specifically, this includes:

- [All journals published by the American Association for the Advancement of Science (AAAS)](https://www.aaas.org/journals)
- [eLife](https://elifesciences.org)

This permission is granted as a special exemption under Section 7 of the _[License](license.md)_.

!!! failure "Closed Access / Author Fee"

    You must obtain a separate license to publish in closed access / author fee venues.

We do not endorse publishing in closed access or author fee venues. To publish in these venues, you must obtain a separate license from us. This typically involves one or more of the following:

- Co-authorship with the DanLing Team.
- Fees to support the project.

While not mandatory, we recommend citing the MultiMolecule project in your research papers.

## 4. Publication in Nature Machine Intelligence

We believe that free and open access to research is a cornerstone of the machine learning community. Inspired by the [Statement on Nature Machine Intelligence](https://openaccess.engineering.oregonstate.edu), and by the ongoing culture of [zero-cost open access](https://diamasproject.eu), we hold that research should be universally accessible without barriers to authors or readers.

[Nature Machine Intelligence](https://www.nature.com/natmachintell/) adopts a closed-access or author-fee model that contradicts these fundamental values. We view such practices as a regressive step in the evolution of machine learning research dissemination, one that undermines community efforts to foster open collaboration and knowledge sharing.

MultiMolecule strictly prohibits the use of its code, data, models, or any related artifacts in works submitted to or published by Nature Machine Intelligence. This prohibition overrides and supersedes any prior waivers, exemptions, or permissions. Unless an additional term explicitly states that it preempts this section, no use of MultiMolecule in Nature Machine Intelligence publications is permitted.

## 5. Can I use MultiMolecule for commercial purposes?

YES! MultiMolecule can be used for commercial purposes under the _[License](license.md)_. Note that you must open-source any modifications to the source code and make them available under the _[License](license.md)_.

If you prefer to use MultiMolecule for commercial purposes without open-sourcing your modifications, you must obtain a separate license from us. This typically involves fees to support the project. Contact us at [multimolecule@zyc.ai](mailto:multimolecule@zyc.ai) for further details.

## 6. Do people affiliated with certain organizations have specific license terms?

YES! If you are affiliated with an organization that has a separate license agreement with us, you may be subject to different license terms.

Members of the following organizations automatically receive a non-redistributable [MIT License](https://mit-license.org/) to use MultiMolecule:

- [Microsoft Research AI for Science](https://www.microsoft.com/en-us/research/lab/microsoft-research-ai-for-science/)
- [DP Technology](https://dp.tech/)

This permission is granted as a special exemption under Section 7 of the _[License](license.md)_. Key points include:

- **Non-Redistributable**: The MIT License granted to members of these organizations is non-transferable and non-sublicensable.
- **No Independent Derivative Works**: You are prohibited from creating independent derivative works based on this license.
- **Compliance with the AGPL License**: Any modifications or derivative works based on this license are automatically considered derivative works of MultiMolecule and must comply with all terms of the _[License](license.md)_. This ensures that third parties cannot circumvent the license terms or establish separate licenses through derivative works.

If you are not affiliated with an organization listed above, please consult your organization's legal department to determine if you are subject to a separate license agreement with us.

## 7. How can I use MultiMolecule if my organization forbids the use of code under the AGPL License?

Certain organizations, such as [Google](https://opensource.google/documentation/reference/using/agpl-policy), prohibit the use of AGPL-licensed code. If you are affiliated with an organization that disallows the use of AGPL-licensed software, you must obtain a separate license from us to use MultiMolecule.

To request a separate license, please contact us at [multimolecule@zyc.ai](mailto:multimolecule@zyc.ai).

## 8. Can I use MultiMolecule if I am a federal employee of the United States Government?

No, federal employees of the United States Government cannot use MultiMolecule under the _[License](license.md)_ because code authored by U.S. federal employees is not protected by copyright under [17 U.S. Code § 105](https://www.law.cornell.edu/uscode/text/17/105).

As a result, federal employees are unable to comply with the terms of the _[License](license.md)_ and, therefore, cannot use MultiMolecule.

## 9. Do we make updates to this FAQ?

!!! tip "In Short"

    Yes, we will update this FAQ as necessary to stay compliant with relevant laws.

We may update this license FAQ from time to time.
The updated version will be indicated by an updated 'Last Revised Time' at the bottom of this license FAQ.
If we make any material changes, we will notify you by posting the new license FAQ on this page.
We are unable to notify you directly as we do not collect any contact information from you.
We encourage you to review this license FAQ frequently to stay informed of how you can use our data, models, code, configuration, and documentation.
