# License FAQ

This License FAQ (Frequently Asked Questions) clarifies the terms and conditions governing the use of the materials in the MultiMolecule project (the "MultiMolecule") provided by the DanLing Team (also known as DanLing) ("we," "us," or "our").
This FAQ serves as an addendum to, and is incorporated by reference into, the [GNU Affero General Public License (AGPL)](license.md) (the "License").
This FAQ and the License together constitute the entire agreement (the "Agreement") between you and us regarding your use of MultiMolecule.
Capitalized terms used but not defined in this FAQ have the meanings given to them in the AGPL.

## 0. Summary of Key Points

This summary highlights the key aspects of our license. For more detailed information, please refer to the corresponding sections below and read the [License](license.md).

<div class="grid cards" markdown>

!!! question "What are source code and object code in MultiMolecule?"

    The Source Code includes all materials necessary to develop, train, evaluate and run a model, including data, code, configuration and documentation. The Object Code include the model weights.

    [:octicons-arrow-right-24: What are source code and object code in MultiMolecule?](#1-what-are-source-code-and-object-code-in-multimolecule)

!!! question "Am I required to share my trained model?"

    Yes, all models must be made available under the terms of the Agreement, along with the Corresponding Source.

    [:octicons-arrow-right-24: Am I required to share my trained model?](#2-am-i-required-to-share-my-trained-model)

!!! question "Am I required to share the data used for training?"

    Yes, all data used for training a model must be made available under the terms of the Agreement.

    [:octicons-arrow-right-24: Am I required to share the data used for training?](#3-am-i-required-to-share-the-data-used-for-training)

!!! question "Do I need to acknowledge MultiMolecule?"

    Yes, you need to acknowledge MultiMolecule if you use it in your research papers or other projects.

    [:octicons-arrow-right-24: Do I need to acknowledge MultiMolecule?](#4-do-i-need-to-acknowledge-multimolecule)

!!! question "Can I publish research papers using MultiMolecule?"

    It depends. You may publish research papers in certain open access and non-profit venues. For closed access or author fee venues, you must obtain a separate license.

    [:octicons-arrow-right-24: Can I publish research papers using MultiMolecule?](#5-can-i-publish-research-papers-using-multimolecule)

!!! question "Is there any restriction on publishing research papers in certain venues?"

    Yes, there are some restrictions on publishing research papers in certain venues.

    [:octicons-arrow-right-24: Restrictions on Publishing Research Papers in Certain Venues](#6-restrictions-on-publishing-research-papers-in-certain-venues)

!!! question "Can I use MultiMolecule for commercial purposes?"

    Yes, you can use MultiMolecule for commercial purposes under the terms of the Agreement.

    [:octicons-arrow-right-24: Can I use MultiMolecule for commercial purposes?](#7-can-i-use-multimolecule-for-commercial-purposes)

!!! question "Are there special permissions for MultiMolecule Collaborators?"

    Yes, recognized Collaborators are granted specific additional permissions pursuant to Section 7 of the License.

    [:octicons-arrow-right-24: Are there special permissions for MultiMolecule Collaborators?](#8-are-there-special-permissions-for-multimolecule-collaborators)

</div>

## 1. What are source code and object code in MultiMolecule?

For all materials in the MultiMolecule project, the following definitions clarify and supplement those found in the [License](license.md):

!!! note "Source Code"

    **Source Code** refers to the preferred form of the licensed materials for making modifications thereto, consistent with Section 1 of the License. It encompasses all materials necessary for developing, training, evaluating and running the models.

Source Code includes, but is not limited to:

- **Data**: The datasets, in the form needed for processing, that are required for training, evaluating, or running the models provided or generated as part of the licensed materials.
- **Code**: All source code for scripts, programs, libraries (including model architecture and pipeline definitions), and utilities required to process data, train models, perform evaluations, deploy the models, or otherwise operate and modify the licensed materials.
- **Configuration**: Configuration files, settings parameters, environmental specifications, and any scripts used to control the installation, compilation, training, evaluation, running, or execution processes related to the licensed materials.
- **Documentation**: Interface definition files, build instructions, manuals, guides, relevant research papers describing the specific methodologies, architectures and parameters used, and any other technical documentation necessary to understand, install, operate, and modify the licensed materials.

*Providing the Source Code as defined here is necessary to satisfy the requirement to provide the "Corresponding Source" under the License when conveying Object Code or providing network access.*

!!! note "Object Code"

    **Object Code** refers to any form of the licensed materials that is not Source Code.

Object Code primarily includes, but is not limited to:

- **Model Weights**: The numerical parameters representing the learned state of a model after training (e.g., files in SafeTensors, HDF5, or similar formats).
- **Compiled Code**: Any executable software code not in human-readable source form, like compiled C++ extensions sometimes found in Python packages.

Understanding this distinction helps clarify your obligations under the AGPL. For instance, if you share or provide network access to Object Code (like a trained model), you must also ensure the corresponding Source Code (including the necessary data, code, configuration, and documentation) is available according to the license terms.

## 2. Am I required to share my trained model?

Yes, you must make your trained model and their weights available under the terms of the Agreement.

As defined in Section 1 of this FAQ, model weights are considered object code in MultiMolecule. Therefore, in addition to making the modified model weights available, you must also provide the Corresponding Source, as defined in Section 1 of the [License](license.md) and expanded in Section 1 of this FAQ, necessary to reproduce the modified model. This includes, but is not limited to, any code, data, models, configuration files, and scripts used in the fine-tuning process, along with the modified model weights themselves.

If you use the work for any purpose, you are obligated to make available the entire project, including code, datasets, and models, under the terms of the Agreement. This approach ensures that all enhancements, modifications, and related resources remain freely available to the community.

## 3. Am I required to share the data used for training?

Yes, you must make the datasets used for that training available under the terms of the Agreement.

As defined in Section 1 of this FAQ, data used for training are considered source code in MultiMolecule. Therefore, it falls under the definition of Corresponding Source required by the [License](license.md). This obligation ensures that others can reproduce or further develop the model using the same data inputs that used for train such models.

This requirement applies specifically to data that was used to train, update, or modify the model weights. Data used _solely_ for the purpose of evaluating the performance of a model (e.g., running inference on a private test set) **does not** need to be made available under this provision, provided it was not also used in any training for that model.

## 4. Do I need to acknowledge MultiMolecule?

Yes, you need to acknowledge MultiMolecule if you use it in your research papers or other projects.

!!! paper "Research Papers"

    Research Papers should contain formal citation to the MultiMolecule project.

Any research paper utilizing the work _shall include a formal citation_ to the MultiMolecule project. The specific citation format may vary depending on the publication venue's style guidelines, but must, at a minimum, include the project name ("MultiMolecule") and the DOI ([10.5281/zenodo.12638419](https://doi.org/10.5281/zenodo.12638419)). If formal citation is not possible due to venue restrictions, the MultiMolecule project name and DOI _shall_ be mentioned in the acknowledgments section.

!!! code "Projects"

    Other projects should contain proper acknowledgement to the MultiMolecule project.

Any other project (including software, web services, and other derivative works) that utilizes MultiMolecule _shall provide proper acknowledgement_ of the MultiMolecule project.

- **Software**: Any software utilizing MultiMolecule must include a prominent display of the MultiMolecule project name and URL in its documentation. Additionally, upon each launch of the resulting executable program (or any program dependent thereon), a prominent display (e.g., a splash screen or banner text) of the MultiMolecule project name and URL _shall_ be presented to the user.
- **Web Services**: Any web service utilizing MultiMolecule must include a prominent display of the MultiMolecule project name and URL on the service's main page or other readily accessible location.
- **Other Derivative Works**: Any other derivative works must include clear acknowledgement.

## 5. Can I publish research papers using MultiMolecule?

Any publication utilizing MultiMolecule, regardless of the publication venue, _shall include a formal citation_ to the MultiMolecule project as described in Section 4.

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

## 6. Restrictions on Publishing Research Papers in Certain Venues

We believe that free and open access to research is a cornerstone of the machine learning community. Inspired by the [Statement on Nature Machine Intelligence](https://openaccess.engineering.oregonstate.edu), and by the ongoing culture of [zero-cost open access](https://diamasproject.eu), we hold that research should be universally accessible without barriers to authors or readers.

The following publication venues adopt closed-access or author-fee models that contradict these fundamental values. We view such practices as a regressive step in the evolution of machine learning research dissemination, one that undermines community efforts to foster open collaboration and knowledge sharing:

- [Nature Machine Intelligence](https://www.nature.com/natmachintell)

We strongly discourage publishing work that uses MultiMolecule in the venues listed above due to their closed-access or author-fee models.

**Notwithstanding any additional permissions granted under Section 7 of the [License](license.md), or through any separate written license agreement with us, publication in the venues listed above is _not authorized_ unless that separate agreement explicitly and specifically states otherwise.** This restriction takes precedence over any other terms in such additional permissions or agreements and supersedes any prior waivers, exemptions, or permissions, unless an additional term explicitly states that it preempts this section.

## 7. Can I use MultiMolecule for commercial purposes?

YES! MultiMolecule can be used for commercial purposes, provided you comply with all terms of the Agreement. This includes the requirement to make the Corresponding Source (as defined in Section 1 of the [License](license.md) and Section 1 of this FAQ) of any modified versions of MultiMolecule, and any related artifacts, including your training data (as clarified in Section 3) and trained models (as clarified in Section 2), available under the terms of the Agreement.

If you prefer to use MultiMolecule for commercial purposes without making your modifications and related materials available under the [License](license.md), you must obtain a separate, written license agreement from us. This may involve financial contributions to support the project. Contact us at [license@danling.org](mailto:license@danling.org) for further details.

## 8. Are there special permissions for MultiMolecule Collaborators?

YES! If you are a Collaborator of the MultiMolecule project (as recognized by the DanLing Team), you are entitled to the following additional permissions granted under Section 7 of the [License](license.md):

!!! tip "Internal Network Use Waiver"
    Collaborators receive a waiver for AGPL Section 13 source sharing obligations related to internal R&D network use involving modified the work.

Notwithstanding Section 13 of the [License](license.md), as a recognized Collaborator, you are not required to offer Corresponding Source to users interacting remotely through a computer network with MultiMolecule (including code, models, or data licensed under this Agreement) that have been modified, _provided that_ such interaction is solely for internal research and development purposes within their respective team.

!!! success "Expanded Permission for Publishing Papers"
    Collaborators receive expanded permission to publish papers compared to Section 5, allowing use of most peer-reviewed venues.

You may publish research papers utilizing MultiMolecule in any peer-reviewed scientific venue (including journals and conference proceedings, regardless of their access model or author fees) or post preprints (e.g., on arXiv). This permission means you are not limited to the specific types of venues or manuscript licenses mentioned in Section 5 and do not require a separate license agreement from us for publication. **Pursuant to Section 6 of this FAQ, this expanded permission does _not_ apply to certain enumerated venues.** Publication in such venues is therefore not covered by the additional permissions granted in this Agreement. You must still comply with the Acknowledgment and Citation requirements described in Section 4.

!!! warning "Source Release Timing Regarding Publications"
    Collaborators receive specific conditions regarding the timing for releasing source code related to publications (acceptance or 366 days post-preprint).

The following conditions apply regarding the timing for providing Corresponding Source for modifications when required by the [License](license.md) (e.g., upon conveyance or for network use not covered by the waiver above): If your modifications are utilized in research described in a manuscript, the obligation under the [License](license.md) to make the Corresponding Source publicly available **is triggered upon** whichever of the following events occurs _first_:

* a) The manuscript's formal _acceptance_ for publication in a peer-reviewed venue (e.g., journal, conference proceedings).
* b) 366 days have passed since the manuscript was first posted on a public preprint server (e.g., arXiv).

You must provide the Corresponding Source immediately upon the occurrence of the first applicable trigger event. If modifications are conveyed or used in ways not tied to a publication or preprint, the standard terms of the [License](license.md) regarding the timing of source provision apply.

**General Conditions Applicable to Collaborator Permissions**:
These permissions are granted only to active, invited Collaborators recognized by the DanLing Team. All other provisions of the [License](license.md) and this FAQ, which are not explicitly modified by these additional permissions, remain in full force and effect. These additional permissions are non-transferable and non-sublicensable.

The DanLing Team may grant additional, specific permissions to Collaborators on a case-by-case basis through written communication.

## 9. How can I use MultiMolecule if my organization forbids the use of code under the AGPL License?

Certain organizations, such as [Google](https://opensource.google/documentation/reference/using/agpl-policy), prohibit the use of AGPL-licensed code. If you are affiliated with an organization that disallows the use of AGPL-licensed software, you must obtain a separate license from us to use MultiMolecule.

To request a separate license, please contact us at [license@danling.org](mailto:license@danling.org).

## 10. Can I use MultiMolecule if I am a federal employee of the United States Government?

Pursuant to [17 U.S. Code § 105](https://www.law.cornell.edu/uscode/text/17/105), the federal employees of the United States Government acting in their official capacity are generally unable to comply with the copyright-based terms of the License and are therefore precluded from utilizing MultiMolecule under this Agreement.

## 11. Do we make updates to this FAQ?

!!! tip "In Short"

    Yes, we will update this FAQ as necessary to stay compliant with relevant laws.

We may update this license FAQ from time to time.
The updated version will be indicated by an updated 'Last Revised Time' at the bottom of this license FAQ.
If we make any material changes, we will notify you by posting the new license FAQ on this page.
We are unable to notify you directly as we do not collect any contact information from you.
We encourage you to review this license FAQ frequently to stay informed of how you can use our data, models, code, configuration, and documentation.
