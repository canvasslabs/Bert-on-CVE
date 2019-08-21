############################################################################
# Copyright 2019 Junliang Yu                                               #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
# http://www.apache.org/licenses/LICENSE-2.0                               #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
############################################################################


def output_pn_vs(tag_register):

    if 'all' in tag_register:
        vs = tag_register['all']
    elif 'version' in tag_register:
        update = tag_register['update'].get('version','') if 'update' in tag_register else ''
        vs = tag_register['version'] + ' ' + update
    elif 'vrs' not in tag_register and 'vre' not in tag_register and 'vr' not in tag_register:
        vs = ''
    elif 'vrs' not in tag_register and 'vre' not in tag_register and 'vr' in tag_register:
        update = tag_register['update'].get('vr','') if 'update' in tag_register else ''
        vs = tag_register['vr'] + ' ' + update
    elif '<' not in tag_register and '<=' not in tag_register and '>' not in tag_register:
        vs = ''
    else:
        if 'vr' in tag_register:
            if '<' in tag_register or '<=' in tag_register:
                if 'vre' not in tag_register:
                    vre = tag_register['vr']
                    vrs = tag_register.get('vrs','')
                else:
                    vre = tag_register['vre']
                    vrs = tag_register.get('vrs',tag_register['vr'])
            elif '>' in tag_register:
                if 'vrs' not in tag_register:
                    vrs = tag_register['vr']
                    vre = tag_register.get('vre','')
                else:
                    vrs = tag_register['vrs']
                    vre = tag_register.get('vre',tag_register['vr'])

            tag_register['vrs'] = vrs
            tag_register['vre'] = vre
        else:
            vrs = tag_register.get('vrs','')
            vre = tag_register.get('vre','')

        update = tag_register['update'].get('vrs','') if 'update' in tag_register else ''
        vrs += ' ' + update
        update = tag_register['update'].get('vre','') if 'update' in tag_register else ''
        vre += ' ' + update
        if '<=' in tag_register:
            vs = '[' + vrs + ',' + vre + ']'
        elif '<' in tag_register:
            if '=' in tag_register:
                vs = '[' + vrs + ' , ' + vre + ']'
            else:
                vs = '[' + vrs + ' , ' + vre + ')'
        elif '>' in tag_register:
            if '=' in tag_register:
                vs = '[' + vrs + ' , ' + vre + ']'
            else:
                vs = '(' + vrs + ' , ' + vre + ')'
    pn = tag_register.get('pn',[])
    return pn, vs, tag_register

def output(pn_list, vs, tag_register, pn_vs_dict):
    if vs!='' and 'sp' not in tag_register:
        done = set()
        for pn in pn_list:
            if pn not in done:
                pn_vs_dict[pn]=pn_vs_dict.get(pn,[])+[vs]
            done.add(pn)
    return pn_vs_dict

def get_version(sequence, tags):
    '''
    Input:
    sequence: list of words
    tags: list of tags

    output:
    a dictionary whose keys are product names and values are list of version ranges
    '''

    pn_vs_dict = {}
    tag_register = {'pn':[]} #record version components by now
    vs_now = None
    for j in range(len(sequence)+1):

        if j==len(sequence):
            pn_list, vs, tag_register = output_pn_vs(tag_register)
            pn_vs_dict = output(pn_list, vs, tag_register, pn_vs_dict)
            break

        if tags[j]!='O':
            tj = tags[j]

            if tj == 'pn':
                pn_list, vs, tag_register = output_pn_vs(tag_register)
                if vs=='':
                    if j==0 or tags[j-1]!='pn':
                        if 'pn' not in tag_register:
                            tag_register['pn'] = []
                        tag_register['pn'].append(sequence[j])
                    else:
                        tag_register['pn'][-1] += ' ' + sequence[j]
                else:
                    pn_vs_dict = output(pn_list, vs, tag_register, pn_vs_dict)
                    tag_register = {'pn':[sequence[j]]}

            elif tj == 'sp':
                pn_list, vs, tag_register = output_pn_vs(tag_register)

                pn_vs_dict = output(pn_list, vs, tag_register, pn_vs_dict)

                tag_register['sp']=True
                if 'all' in tag_register:
                    tag_register.pop('all')


            elif tj=='update':
                vs_now = tag_register.get('vs_now','')
                if '<' in tag_register or '<=' in tag_register:
                    if 'vre' not in tag_register:
                        tag_register['vre'] = tag_register['vrs']
                        vs_now = 'vre'
                elif '>' in tag_register:
                    if 'vrs' not in tag_register:
                        tag_register['vrs'] = tag_register['vre']
                        vs_now = 'vrs'

                if 'update' not in tag_register:
                    tag_register['update'] = {}
                if vs_now not in tag_register['update']:
                    tag_register['update'][vs_now] = sequence[j]
                    if 'sp' in tag_register:
                        tag_register.pop('sp')
                else:
                    tag_register['update'][vs_now] += ' ' + sequence[j]

            else:
                if 'sp' in tag_register:
                    tag_register = {'pn':tag_register['pn']}
                tag_register[tj] = sequence[j]

                if tj in ['version','vr','vrs','vre']:
                    if 'vs_now' in tag_register and tag_register['vs_now']=='':
                        tag_register['update'][tj] = tag_register['update']['']
                        tag_register['update'].pop('')
                    tag_register['vs_now'] = tj
    return pn_vs_dict
