# License FAQ

This License FAQ (Frequently Asked Questions) clarifies the terms and conditions governing the use of the materials in the MultiMolecule project (the "MultiMolecule") provided by the DanLing Team (also known as DanLing) ("we," "us," or "our").
This FAQ serves as an addendum to, and is incorporated by reference into, the [GNU Affero General Public License (AGPL)](license.md) (the "License").
This FAQ and the License together constitute the entire agreement (the "Agreement") between you and us regarding your use of MultiMolecule.
Capitalized terms used but not defined in this FAQ have the meanings given to them in the AGPL.

## 0. Summary of Key Points

This summary highlights the key aspects of our license.
For more detailed information, please refer to the corresponding sections below and read the [License](license.md).

<div class="grid cards" markdown>

!!! question "What are source code and object code in MultiMolecule?"

    The Source Code includes all materials necessary to develop, train, evaluate, and run a model, including data, code, configuration, and documentation.
    The Object Code includes model weight files and compiled code.

    [:octicons-arrow-right-24: What are source code and object code in MultiMolecule?](#1-what-are-source-code-and-object-code-in-multimolecule)

!!! question "Am I required to share my trained model?"

    If you Convey model weight files hosted and distributed by MultiMolecule, you must convey those weight files under the Agreement, along with the Corresponding Source.
    If you Convey modified versions of such weight files (for example, fine-tuned weights), you must convey those modified weight files under the Agreement, along with the Corresponding Source.
    These obligations apply regardless of whether you used MultiMolecule, a third-party library, or a customized training pipeline to produce the conveyed weights.

    [:octicons-arrow-right-24: Am I required to share my trained model?](#2-am-i-required-to-share-my-trained-model)

!!! question "Am I required to share the data used for training?"

    If you Convey any model weights covered by Section 2, you must also provide to recipients under the Agreement any training datasets used to train, update, or modify those weights, excluding data used solely for evaluation as clarified in Section 3.

    [:octicons-arrow-right-24: Am I required to share the data used for training?](#3-am-i-required-to-share-the-data-used-for-training)

!!! question "Do I need to acknowledge MultiMolecule?"

    We strongly encourage acknowledgement whenever MultiMolecule contributes to your work.
    Citation is strongly requested for all research papers and becomes mandatory only as a condition of additional permissions granted in Section 5 or Section 8.
    Reasonable author attribution in Appropriate Legal Notices is required in certain distributed interactive interfaces.

    [:octicons-arrow-right-24: Do I need to acknowledge MultiMolecule?](#4-do-i-need-to-acknowledge-multimolecule)

!!! question "Can I publish research papers using MultiMolecule?"

    If your manuscript or supplements include MultiMolecule materials (as described in Section 5), then the manuscript and supplements must be distributed under the License unless an additional permission applies.
    Section 5 and Section 8 grant additional permissions to enable publication in certain venues under manuscript-sharing licenses, subject to Section 6.
    If you cannot comply with the License and do not qualify for an additional permission, you may not distribute the manuscript or supplements with MultiMolecule materials.

    [:octicons-arrow-right-24: Can I publish research papers using MultiMolecule?](#5-can-i-publish-research-papers-using-multimolecule)

!!! question "Is there any restriction on publishing research papers in certain venues?"

    Yes, there are restrictions on publishing research papers in certain venues under the additional permissions granted by this FAQ.
    Section 6 limits only additional permissions granted by this FAQ.
    Section 6 does not restrict publication under the [License](license.md) itself.

    [:octicons-arrow-right-24: Restrictions on Publishing Research Papers in Certain Venues](#6-restrictions-on-publishing-research-papers-in-certain-venues)

!!! question "Can I use MultiMolecule for commercial purposes?"

    Yes, you can use MultiMolecule for commercial purposes under the terms of the Agreement.
    If you prefer commercial use without the obligations that apply when you Convey covered materials, you must obtain a separate license.

    [:octicons-arrow-right-24: Can I use MultiMolecule for commercial purposes?](#7-can-i-use-multimolecule-for-commercial-purposes)

!!! question "Are there special permissions for MultiMolecule Collaborators?"

    Yes, recognized Collaborators are granted specific additional permissions pursuant to Section 7 of the License.
    These permissions are subject to the stated conditions and to Section 6.

    [:octicons-arrow-right-24: Are there special permissions for MultiMolecule Collaborators?](#8-are-there-special-permissions-for-multimolecule-collaborators)

</div>

## 1. What are source code and object code in MultiMolecule?

For all materials in the MultiMolecule project, the following definitions clarify and supplement those found in the [License](license.md).

> [!TIP] Scope of materials hosted by MultiMolecule
> Unless explicitly stated otherwise in the relevant model card, dataset card, file header, directory notice, or accompanying LICENSE/NOTICE, all model weights, datasets, code, configuration, and documentation hosted and distributed by MultiMolecule are provided under the Agreement.
> If we host any specific item under different terms in the future, we will explicitly label that item, and the stated terms will control for that item.

> [!IMPORTANT] Source Code
> **Source Code** refers to the preferred form of the licensed materials for making modifications thereto, consistent with Section 1 of the License.
> It encompasses all materials necessary for developing, training, evaluating and running the models.

Source Code includes, but is not limited to:

- **Data**: The datasets, in the form needed for processing, that are required for training, evaluating, or running the models provided or generated as part of the licensed materials.
- **Code**: All source code for scripts, programs, libraries (including model architecture and pipeline definitions), and utilities required to process data, train models, perform evaluations, deploy the models, or otherwise operate and modify the licensed materials.
- **Configuration**: Configuration files, settings parameters, environmental specifications, and any scripts used to control the installation, compilation, training, evaluation, running, or execution processes related to the licensed materials.
- **Documentation**: Interface definition files, build instructions, manuals, guides, research papers and technical reports distributed by MultiMolecule as part of the licensed materials describing the specific methodologies, architectures and parameters used, and any other technical documentation necessary to understand, install, operate, and modify the licensed materials.

*Providing the Source Code as defined here is necessary to satisfy the requirement to provide the "Corresponding Source" under the Agreement (and where applicable, under the License) when conveying Object Code.*

> [!IMPORTANT] Object Code
> **Object Code** refers to any form of the licensed materials that is not Source Code.

Object Code primarily includes, but is not limited to:

- **Model Weights**: The numerical parameters representing the learned state of a model after training (e.g., files in SafeTensors, HDF5, or similar formats).
  This includes all model weights provided or hosted by MultiMolecule except for those stated otherwise.
  This also includes any fine-tuned model weights derived from model weights provided or hosted by MultiMolecule.
- **Compiled Code**: Any executable software code not in human-readable source form, like compiled C++ extensions sometimes found in Python packages.

*For model weights treated as Object Code in MultiMolecule, the Corresponding Source includes, at a minimum, the training data and the scripts needed to reproduce the conveyed weights.*

Understanding this distinction helps clarify your obligations under the Agreement.
For instance, if you Convey Object Code (like model weights), you must also ensure the corresponding Source Code (including the necessary data, code, configuration, and documentation) is available under the terms of the Agreement.

## 2. Am I required to share my trained model?

If you Convey model weights covered by the Agreement, you must convey those weights under the Agreement, along with the Corresponding Source.

As explained in Section 1 of this FAQ, model weights are treated as Object Code in MultiMolecule.
Whenever you Convey Object Code under the Agreement, you must also provide the Corresponding Source.
For model weights covered by the Agreement, Corresponding Source includes, at a minimum, the code, training data, configuration, and scripts needed to reproduce, install, run, and modify the conveyed weights, as clarified in Section 1 of this FAQ.

If you modify MultiMolecule and you provide users remote interaction with your modified version through a computer network, you must comply with Section 13 of the License by offering the Corresponding Source of your modified version to those users.
This Section 13 obligation concerns remote interaction with the modified MultiMolecule Program itself.

Section 3 specifies the training-data requirement.

## 3. Am I required to share the data used for training?

If you Convey model weights covered by the Agreement, you must also provide to recipients under the Agreement any training datasets used to train, update, or modify those weights, excluding data used solely for evaluation as clarified below.

As explained in Section 1 of this FAQ, training datasets are treated as Source Code in MultiMolecule.
Accordingly, whenever you are required to provide Corresponding Source under the Agreement, the required Corresponding Source includes the training data required to reproduce the conveyed weights, as clarified in Section 1 of this FAQ.

This requirement applies only to data used to train, update, or modify the conveyed model weights.
Data used solely for evaluation is not required under this provision, provided it was not also used for training.

If the required training data cannot be provided to recipients under the Agreement, you may not Convey the resulting weights under the Agreement.

## 4. Do I need to acknowledge MultiMolecule?

We strongly encourage acknowledgement whenever MultiMolecule contributes to your work.
This section distinguishes

- (a) what we strongly request as a community norm, and
- (b) what becomes mandatory as a condition of additional permissions granted by this FAQ.

> [!NOTE]
> We strongly encourage formal citation in any research paper that uses MultiMolecule.
> If you publish a paper solely under the [License](license.md), citation is not a condition of license compliance, but it is strongly requested.
> If you rely on an additional permission in Section 5 or Section 8, formal citation is a condition of that additional permission.

When citation is required under this FAQ (e.g., as a condition of an additional permission in Section 5 or Section 8), it must include, at a minimum, the project name (“MultiMolecule”) and the DOI ([10.5281/zenodo.12638419](https://doi.org/10.5281/zenodo.12638419)).
If the venue does not support formal citations, the project name and DOI must instead appear in the acknowledgments section.

> [!IMPORTANT]
> If you Convey a program that incorporates MultiMolecule, you must preserve a reasonable author attribution for MultiMolecule in the Appropriate Legal Notices.

If the Program has an interactive user interface, it must display Appropriate Legal Notices.
Those notices must include a reasonable author attribution for MultiMolecule, including the project name and a link to the official repository or website.

For command-line programs, this attribution must be shown prominently at startup and be available via `--help`, `--version`, or `--about` where applicable.
For web services, this attribution must be shown prominently on the main page or another readily accessible location.

For libraries or non-interactive components, this attribution must be shown prominently in the documentation and, if the component provides any interactive interface, in that interface.

## 5. Can I publish research papers using MultiMolecule?

> [!IMPORTANT]
> As clarified in Section 1, Documentation is part of Source Code in MultiMolecule.
> Accordingly, if your manuscript or supplements include MultiMolecule materials (i.e., contain or reproduce MultiMolecule code, weights, datasets, documentation text, figures, or other MultiMolecule materials), then the manuscript and supplements are treated as Documentation distributed with those MultiMolecule materials.
> This requirement concerns manuscripts and supplements that include MultiMolecule materials, not separate and independent works that are merely distributed alongside MultiMolecule.
> Therefore, absent an additional permission, the manuscript and supplements that include MultiMolecule materials must be distributed under the License to the extent the manuscript or supplements contain or reproduce MultiMolecule materials, as part of the same distribution of those MultiMolecule materials.
> If you cannot comply with the License, you may not distribute the manuscript or supplements with MultiMolecule materials under the Agreement.

This section grants additional permissions under Section 7 of the License for specific publication scenarios in which authors prefer to release manuscripts under manuscript-sharing licenses.
For avoidance of doubt, the additional permissions in this Section 5 apply only to the manuscript text and other documentary materials.
They do not alter the License that governs any MultiMolecule code, model weights, or datasets that you Convey, which remain under the Agreement unless explicitly stated otherwise.


> [!IMPORTANT]
> If you rely on any additional permission in this Section 5, formal citation as described in Section 4 is a condition of that additional permission.
> Any additional permission granted in this Section 5 remains subject to Section 6.

> [!TIP] Diamond Open Access
> Diamond open access venues are permitted under the additional permissions below.

You may publish manuscripts that Convey MultiMolecule materials in fully open access journals, conferences, or platforms that do not charge fees to either authors or readers.

The public version of the manuscript must be made available under a license that permits sharing of manuscripts.
You may use one of the following licenses.

- GNU Free Documentation License (GFDL)
- Creative Commons licenses
- OSI-approved licenses

This permission is granted as an additional permission under Section 7 of the [License](license.md).

> [!WARNING] Non-Profit
> Certain non-profit venues are permitted under the additional permissions below.

You may publish manuscripts that Convey MultiMolecule materials in certain non-profit journals, conferences, or platforms.

This includes the following venues.

- eLife

This permission is granted as an additional permission under Section 7 of the [License](license.md).

> [!CAUTION] Closed-Access / Author-Fee
> Closed-access or author-fee venues often make compliance impossible.

We do not endorse publishing MultiMolecule materials in closed-access or author-fee venues.

If a venue’s terms would prevent you from complying with the License for any MultiMolecule materials you Convey in connection with the publication, you must obtain a separate written license agreement from us prior to submission or publication.
Such an agreement may involve conditions such as co-authorship or financial contributions to the project.

## 6. Restrictions on Publishing Research Papers in Certain Venues

> [!IMPORTANT]
> This section limits only additional permissions granted by this FAQ.
> This section does not restrict publication under the [License](license.md) itself.
> If you Convey MultiMolecule materials as part of a publication and you comply with the License for those materials, Section 6 does not apply.

Accordingly, Section 6 constrains only the additional permissions in Section 5 and Section 8.
Section 6 also constrains any separate written license agreement that incorporates or references this FAQ, unless that separate agreement expressly states otherwise in writing.

We believe that free and open access to research is a cornerstone of the machine learning community.
Inspired by the [Statement on Nature Machine Intelligence](https://openaccess.engineering.oregonstate.edu), and by the ongoing culture of [zero-cost open access](https://diamasproject.eu), we hold that research should be universally accessible without barriers to authors or readers.

The following publication venues adopt closed-access or author-fee models that contradict these fundamental values.
We view such practices as a regressive step in the evolution of machine learning research dissemination, one that undermines community efforts to foster open collaboration and knowledge sharing.

- Nature Machine Intelligence

Notwithstanding Sections 5 and 8, none of the additional permissions granted by this FAQ authorize submission or publication of a manuscript that Conveys MultiMolecule materials in the venues listed above.
You may submit or publish such a manuscript in the venues listed above only if you have a separate written license agreement from us that expressly permits publication notwithstanding this Section 6.

We strongly discourage publishing work that Conveys MultiMolecule materials in the venues listed above.

## 7. Can I use MultiMolecule for commercial purposes?

Yes.
You may use MultiMolecule for commercial purposes, provided you comply with the Agreement.

If you Convey modified MultiMolecule materials, you must provide the Corresponding Source and related artifacts required by the License and this FAQ.

Where applicable, this includes training data as clarified in Section 3 and model weights as clarified in Section 2.

If you prefer commercial use without making such materials available under the License, you must obtain a separate written license agreement from us.
Please contact [license@danling.org](mailto:license@danling.org) for details.

## 8. Are there special permissions for MultiMolecule Collaborators?

Yes.
If you are recognized as a Collaborator by the DanLing Team, you are entitled to the following additional permissions granted under Section 7 of the [License](license.md).

> [!TIP] Internal network use waiver
> Notwithstanding Section 13 of the License, Collaborators receive a waiver of the obligation to offer Corresponding Source to users interacting remotely through a computer network with a modified version of MultiMolecule, provided that the interaction is solely for internal research and development within the Collaborator’s team.
> This waiver does not apply to external users, public deployments, or Conveyance.

> [!TIP] Expanded permission for publishing papers
> Collaborators may publish manuscripts that Convey MultiMolecule materials in any peer-reviewed scientific venue, including journals and conference proceedings, regardless of access model or author fees.
> This expanded permission is granted as an additional permission under Section 7 of the [License](license.md).
> This expanded permission remains subject to Section 6 of this FAQ.
> This expanded permission affects only the licensing of the manuscript and supplementary documentation, and does not alter the License that governs any MultiMolecule materials you Convey.
> As a condition of this expanded permission, you must comply with the acknowledgement and citation requirements in Section 4.

> [!IMPORTANT] Source release timing related to publications
> This additional permission concerns the timing of *public release* of publication-related modifications.
> It does not delay any obligation under the License to provide Corresponding Source to recipients upon Conveyance, or to remote users upon network interaction under Section 13.
> If your modifications are utilized in research described in a manuscript, you must make the Corresponding Source for those publication-related modifications publicly available upon the first of the following events.
>
> - The manuscript’s formal acceptance for publication in a peer-reviewed venue.
> - 366 days have passed since the manuscript was first posted on a public preprint server.
>
> You must make the public release immediately upon the first applicable trigger event.
> If modifications are Conveyed or made available for remote interaction through a computer network in ways not tied to a publication or preprint, the standard timing rules of the License apply.

> [!NOTE] General conditions for Collaborator permissions
> These permissions are granted only to active, invited Collaborators recognized by the DanLing Team.
> These permissions are non-transferable and non-sublicensable.
> All other provisions of the License and this FAQ remain in full force and effect unless explicitly modified above.
> The DanLing Team may grant additional case-specific permissions through written communication.

## 9. How can I use MultiMolecule if my organization forbids the use of code under the AGPL License?

Certain organizations, such as [Google](https://opensource.google/documentation/reference/using/agpl-policy), prohibit the use of AGPL-licensed code.
If you are affiliated with an organization that disallows the use of AGPL-licensed software, you must obtain a separate license from us to use MultiMolecule.

To request a separate license, please contact us at [license@danling.org](mailto:license@danling.org).

## 10. Do we make updates to this FAQ?

> [!TIP] "In Short"
> Yes, we will update this FAQ as necessary to stay compliant with relevant laws.

We may update this license FAQ from time to time.
The updated version will be indicated by an updated 'Last Revised Time' at the bottom of this license FAQ.
If we make any material changes, we will notify you by posting the new license FAQ on this page.
We are unable to notify you directly as we do not collect any contact information from you.
We encourage you to review this license FAQ frequently to stay informed of how you can use our data, models, code, configuration, and documentation.
