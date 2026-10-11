// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Project } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Item, Picker, Section } from '@geti-ui/ui';

import { ALL_PROJECTS } from '../utils';

type ProjectFilterPickerProps = {
    projects: Project[];
    value: string;
    onChange: (projectId: string) => void;
    currentProjectId?: string;
};

export const ProjectFilterPicker = ({ projects, value, onChange, currentProjectId }: ProjectFilterPickerProps) => {
    const { t } = useTranslation();

    return (
        <Picker
            label={t('jobs.projectsSection')}
            labelPosition={'side'}
            aria-label='Filter jobs by project'
            width={'size-3000'}
            selectedKey={value}
            onSelectionChange={(key) => onChange(String(key))}
        >
            <Section>
                <Item key={ALL_PROJECTS}>{t('jobs.allProjects')}</Item>
            </Section>
            <Section title={t('jobs.projectsSection')}>
                {projects.map((project) => (
                    <Item key={project.id} textValue={project.name}>
                        {project.id === currentProjectId
                            ? t('jobs.currentProjectLabel', { name: project.name })
                            : project.name}
                    </Item>
                ))}
            </Section>
        </Picker>
    );
};
