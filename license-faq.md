# License FAQ

This License FAQ (Frequently Asked Questions) clarifies the terms and conditions governing the use of data, models, code, configurations, and documentation in the MultiMolecule project (the "MultiMolecule") provided by the DanLing Team (also known as DanLing) ("we," "us," or "our").
This FAQ serves as an addendum to, and is incorporated by reference into, the [GNU Affero General Public License (AGPL)](license.md) (the "License").
This FAQ and the License together constitute the entire agreement (the "Agreement") between you and us regarding your use of MultiMolecule.
Capitalized terms used but not defined in this FAQ have the meanings given to them in the AGPL.

## 0. Summary of Key Points

This summary highlights the key aspects of our license. For more detailed information, please refer to the corresponding sections below and read the [License](license.md).

<div class="grid cards" markdown>

!!! question "What constitutes the 'source code' in MultiMolecule?"

    The source code in MultiMolecule includes all materials necessary for training the models, such as data, code, configuration files, documentation, and research papers. Model weights are considered object code.

    [:octicons-arrow-right-24: What constitutes the 'source code' in MultiMolecule?](#1-what-constitutes-the-source-code-in-multimolecule)

!!! question "Am I required to share my fine-tuned model weights?"

    Yes, any fine-tuned model weights derived from MultiMolecule's pre-trained models must be made available under the terms of the Agreement.

    [:octicons-arrow-right-24: Am I required to share my fine-tuned model weights?](#2-am-i-required-to-share-my-fine-tuned-model-weights)

!!! question "Do I need to acknowledge MultiMolecule?"

    Yes, you need to acknowledge MultiMolecule if you use it in your research papers or other projects.

    [:octicons-arrow-right-24: Do I need to acknowledge MultiMolecule?](#3-do-i-need-to-acknowledge-multimolecule)

!!! question "Can I publish research papers using MultiMolecule?"

    It depends. You may publish research papers in certain open access and non-profit venues. For closed access or author fee venues, you must obtain a separate license.

    [:octicons-arrow-right-24: Can I publish research papers using MultiMolecule?](#4-can-i-publish-research-papers-using-multimolecule)

!!! question "Is there any restriction on publishing research papers in certain venues?"

    Yes, there are some restrictions on publishing research papers in certain venues.

    [:octicons-arrow-right-24: Restrictions on Publishing Research Papers in Certain Venues](#5-restrictions-on-publishing-research-papers-in-certain-venues)

!!! question "Can I use MultiMolecule for commercial purposes?"

    Yes, you can use MultiMolecule for commercial purposes under the terms of the Agreement.

    [:octicons-arrow-right-24: Can I use MultiMolecule for commercial purposes?](#6-can-i-use-multimolecule-for-commercial-purposes)

</div>

## 1. What constitutes the "source code" in MultiMolecule?

In the context of MultiMolecule, "source code" refers to all materials necessary for developing, training, and reproducing the models. This includes, but is not limited to:

- **Data**: Datasets used for training and evaluating the models.
- **Code**: All scripts, programs, and utilities required for model training, evaluation, and deployment.
- **Configuration**: Configuration files and settings that dictate the behavior of the models and training processes.
- **Documentation**: Manuals, guides, and other forms of documentation that facilitate understanding and using MultiMolecule.
- **Research Papers**: Manuscripts and publications that describe the methodologies and findings related to MultiMolecule.

Conversely, the **model weights** generated from the training process are considered "object code."

This distinction ensures that all necessary components for training and modifying the models remain accessible, while the trained model weights are treated as object code, aligning with our commitment to providing maximum freedom for the community.

## 2. Am I required to share my fine-tuned model?

Yes, if you utilize a pre-trained model from MultiMolecule's model hub and subsequently fine-tune it, you must open-source your modified model weights under the terms of the Agreement.

Model weights are considered object code in MultiMolecule. Therefore, in addition to making the modified model weights available, you must also provide the Corresponding Source, as defined in Section 1 of the [License](license.md) and expanded in Section 1 of this FAQ, necessary to reproduce the modified model. This includes, but is not limited to, any code, data, models, configuration files, and scripts used in the fine-tuning process, along with the modified model weights themselves.

In addition, if you use our Corresponding Source for any purpose, you are obligated to make available all related material, including modified code, derived datasets, and fine-tuned models, under the terms of the Agreement. This approach ensures that all enhancements, modifications, and related resources remain freely available to the community.

## 3. Do I need to acknowledge MultiMolecule?

Yes, you need to acknowledge MultiMolecule if you use it in your research papers or other projects.

!!! paper "Research Papers"

    Any research paper utilizing MultiMolecule resources _shall include a formal citation_ to the MultiMolecule project. The specific citation format may vary depending on the publication venue's style guidelines, but must, at a minimum, include the project name ("MultiMolecule") and a link to the project's official website ([https://multimolecule.danling.org](https://multimolecule.danling.org)). If formal citation is not possible due to venue restrictions, the MultiMolecule project name and URL _shall_ be mentioned in the acknowledgments section.

!!! code "Projects"

    Any other project (including software, web services, and other derivative works) that utilizes MultiMolecule components _shall provide proper acknowledgement_ of the MultiMolecule project.

    - **Software**: Any software utilizing MultiMolecule components must include a prominent display of the MultiMolecule project name and URL in its documentation. Additionally, upon each launch of the resulting executable program (or any program dependent thereon), a prominent display (e.g., a splash screen or banner text) of the MultiMolecule project name and URL _shall_ be presented to the user.
    - **Web Services**: Any web service utilizing MultiMolecule components must include a prominent display of the MultiMolecule project name and URL on the service's main page or other readily accessible location.
    - **Other Derivative Works**: Any other derivative works must include clear acknowledgement.

## 4. Can I publish research papers using MultiMolecule?

Any publication utilizing MultiMolecule, regardless of the publication venue, _shall include a formal citation_ to the MultiMolecule project.

!!! success "Open Access"

    You may publish research papers in diamond open access venues.

You are permitted to publish research papers in fully open access journals, conferences, or platforms that do not charge fees to either authors or readers, provided that all published manuscripts are made available under one of the following licenses that permits the sharing of manuscripts:

- [GNU Free Documentation License (GFDL)](https://www.gnu.org/licenses/fdl.html)
- [Creative Commons Licenses](https://creativecommons.org/licenses)
- [OSI-Approved Licenses](https://opensource.org/licenses)

This permission is granted as a additional permission under Section 7 of the [License](license.md).

!!! warning "Non-Profit"

    You may publish research papers in certain non-profit venues.

You are permitted to publish research papers in certain non-profit journals, conferences, or platforms. Specifically, this includes:

- [All journals published by the American Association for the Advancement of Science (AAAS)](https://www.aaas.org/journals)
- [eLife](https://elifesciences.org)

This permission is granted as a additional permission under Section 7 of the [License](license.md).

!!! failure "Closed Access / Author Fee"

    You must obtain a separate license to publish in closed access / author fee venues.

We do not endorse publishing in closed access or author fee venues. Publishing in closed access or author-fee venues requires a separate, written license agreement from us _prior_ to submission or publication. Such an agreement may involve conditions such as co-authorship or financial contributions to the project.

## 5. Restrictions on Publishing Research Papers in Certain Venues

We believe that free and open access to research is a cornerstone of the machine learning community. Inspired by the [Statement on Nature Machine Intelligence](https://openaccess.engineering.oregonstate.edu), and by the ongoing culture of [zero-cost open access](https://diamasproject.eu), we hold that research should be universally accessible without barriers to authors or readers.

The following publication venues adopt closed-access or author-fee models that contradict these fundamental values. We view such practices as a regressive step in the evolution of machine learning research dissemination, one that undermines community efforts to foster open collaboration and knowledge sharing:

- [Nature Machine Intelligence](https://www.nature.com/natmachintell)

We strongly discourage publishing work that uses MultiMolecule in the venues listed above due to their closed-access or author-fee models.

**Notwithstanding any additional permissions granted under Section 7 of the [License](license.md), or through any separate written license agreement with us, publication in the venues listed above is _not authorized_ unless that separate agreement explicitly and specifically states otherwise.** This restriction takes precedence over any other terms in such additional permissions or agreements and supersedes any prior waivers, exemptions, or permissions, unless an additional term explicitly states that it preempts this section.

## 6. Can I use MultiMolecule for commercial purposes?

YES! MultiMolecule can be used for commercial purposes, provided you comply with all terms of the Agreement. This includes the requirement to make the Corresponding Source (as defined in Section 1 of the [License](license.md) and Section 1 of this FAQ) of any modified versions of MultiMolecule, and any related artifacts, including your training data and trained models, available under the terms of the Agreement.

If you prefer to use MultiMolecule for commercial purposes without making your modifications available under the [License](license.md), you must obtain a separate, written license agreement from us. This may involve financial contributions to support the project. Contact us at [multimolecule@zyc.ai](mailto:multimolecule@zyc.ai) for further details.

## 7. Do people affiliated with certain organizations have specific license terms?

YES! If you are affiliated with an organization that has a separate license agreement with us, you may be subject to different license terms.

For individuals affiliated with the following organizations:

- [Microsoft Research AI for Science](https://www.microsoft.com/en-us/research/lab/microsoft-research-ai-for-science)
- [DP Technology](https://dp.tech)

the following additional permission is granted under Section 7 of the [License](license.md):

Notwithstanding Section 13 of the [License](license.md), individuals affiliated with the listed organizations are not required to offer Corresponding Source to users interacting with a modified version of MultiMolecule remotely through a computer network, _provided that_ such use is solely for internal research and development purposes within their respective organization. All other provisions of the [License](license.md), including, but not limited to, the requirement to provide Corresponding Source if you distribute the MultiMolecule and its derivatives outside your organization, remain in full force and effect. This additional permission is non-transferable and non-sublicensable.

If you are _not_ affiliated with one of the listed organizations, these additional permissions do _not_ apply to you, and you must comply with all terms of the [License](license.md), including Section 13. You may wish to consult your organization's legal department to determine if a separate agreement exists.

## 8. How can I use MultiMolecule if my organization forbids the use of code under the AGPL License?

Certain organizations, such as [Google](https://opensource.google/documentation/reference/using/agpl-policy), prohibit the use of AGPL-licensed code. If you are affiliated with an organization that disallows the use of AGPL-licensed software, you must obtain a separate license from us to use MultiMolecule.

To request a separate license, please contact us at [multimolecule@zyc.ai](mailto:multimolecule@zyc.ai).

## 9. Can I use MultiMolecule if I am a federal employee of the United States Government?

No, federal employees of the United States Government cannot use MultiMolecule under the terms of the Agreement because code authored by U.S. federal employees in their official capacity are not subject to copyright protection under [17 U.S. Code § 105](https://www.law.cornell.edu/uscode/text/17/105).
Consequently, federal employees may be unable to comply with the terms of the Agreement.

## 10. Do we make updates to this FAQ?

!!! tip "In Short"

    Yes, we will update this FAQ as necessary to stay compliant with relevant laws.

We may update this license FAQ from time to time.
The updated version will be indicated by an updated 'Last Revised Time' at the bottom of this license FAQ.
If we make any material changes, we will notify you by posting the new license FAQ on this page.
We are unable to notify you directly as we do not collect any contact information from you.
We encourage you to review this license FAQ frequently to stay informed of how you can use our data, models, code, configuration, and documentation.
